#include "ikaros.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

using namespace ikaros;

// Patterns are made in the layout of the epi-head widget, seen from the front: eye LED i sits at
// i * 30 degrees clockwise from 3 o'clock (9 is the top), and mouth LED i counts from the viewer's
// left. The robot's left eye appears on the viewer's right. The outputs for FadeCandy map this layout
// to the wiring of each robot.
class EpiLightShow : public Module
{
    static constexpr int eyeLeds = 12;
    static constexpr int mouthLeds = 8;
    static constexpr int waveRows = 9; // seven eye rows, the upper mouth row, and the lower mouth row

    enum Pattern
    {
        none = 0,
        wave = 1,
        circle = 2,
        check = 3
    };

    struct Orientation
    {
        float firstAngle[2] = {0, 0}; // left eye, right eye: angle of LED 0, clockwise from 3 o'clock
        float direction[2] = {1, 1};  // 1 when the LEDs are numbered clockwise as seen from the front
        bool swapRows = false;        // MOUTH_HIGH drives the lower row
        bool reverse = false;         // mouth LED 0 is on the viewer's right
    };

    parameter robot;
    parameter calibrationFile;
    parameter checkAtStart;
    matrix showPatterns;
    parameter patternTime;
    parameter buttonDuration;
    parameter colorMode;
    parameter brightness;
    parameter waveRate;
    parameter circleRate;
    parameter tail;
    parameter fadeRate;
    parameter seed;

    matrix eyeLeft;
    matrix eyeRight;
    matrix mouthTop;
    matrix mouthBottom;
    matrix run;
    matrix interrupt;
    matrix arousal;
    matrix color;

    matrix leftEye;
    matrix rightEye;
    matrix mouthHigh;
    matrix mouthLow;
    matrix viewLeftEye;
    matrix viewRightEye;
    matrix viewMouthHigh;
    matrix viewMouthLow;
    matrix patternOutput;
    matrix calibrated;
    matrix orientationOutput;

    // Pattern brightness in the widget layout, 0..1 per LED.
    float leftLevel[eyeLeds] = {};
    float rightLevel[eyeLeds] = {};
    float highLevel[mouthLeds] = {};
    float lowLevel[mouthLeds] = {};

    Orientation orientation;
    std::mt19937 randomGenerator;
    std::mutex commandMutex;
    std::vector<std::pair<std::string, int>> commands;

    bool checking = false;
    int pattern = none;
    int showIndex = 0;
    float phase = 0;         // rows of the wave or laps of the circle since the pattern started
    float patternClock = 0;  // seconds since the pattern started
    float buttonLeft = 0;    // seconds left of a pattern from the play command
    float mix = 0;           // 0 shows the expression, 1 the pattern
    float hue = 0.6f;

    static void
    HueToColor(float h, float rgb[3])
    {
        h = h - std::floor(h);
        for(int c = 0; c < 3; ++c)
        {
            const float k = std::fmod(5.0f - 2.0f * c + h * 6.0f, 6.0f);
            rgb[c] = 1.0f - std::max(0.0f, std::min({k, 4.0f - k, 1.0f}));
        }
    }


    float
    Uniform(float low, float high)
    {
        return std::uniform_real_distribution<float>(low, high)(randomGenerator);
    }


    float
    Arousal() const
    {
        return arousal.connected() ? std::clamp(arousal(0), 0.0f, 1.0f) : 0.3f;
    }


    std::filesystem::path
    CalibrationPath(bool forWriting)
    {
        std::filesystem::path path;
        const bool ok = forWriting ? kernel().SanitizeWritePath(calibrationFile.as_string(), path)
                                   : kernel().SanitizeReadPath(calibrationFile.as_string(), path);
        return ok ? path : std::filesystem::path();
    }


    bool
    LoadOrientation()
    {
        const std::filesystem::path path = CalibrationPath(false);
        if(path.empty() || !std::filesystem::exists(path))
            return false;
        dictionary file;
        try
        {
            file.load_json(path.string());
        }
        catch(const std::exception & error)
        {
            Notify(msg_warning, "EpiLightShow could not read " + path.string() + ": " + error.what());
            return false;
        }
        const std::string name = robot.as_string();
        if(!file.contains(name) || !file.at(name).is_dictionary())
            return false;
        const dictionary & entry = file.at(name).as_dictionary();
        auto number = [&](const char * key, float fallback)
        {
            return entry.contains(key) ? entry.at(key).as_float() : fallback;
        };
        orientation.firstAngle[0] = number("left_eye_first_angle", 0);
        orientation.direction[0] = number("left_eye_direction", 1) < 0 ? -1.0f : 1.0f;
        orientation.firstAngle[1] = number("right_eye_first_angle", 0);
        orientation.direction[1] = number("right_eye_direction", 1) < 0 ? -1.0f : 1.0f;
        orientation.swapRows = number("mouth_swap_rows", 0) > 0.5f;
        orientation.reverse = number("mouth_reverse", 0) > 0.5f;
        return true;
    }


    // Writes the orientation of this robot into the file and keeps the entries of other robots.
    bool
    SaveOrientation()
    {
        const std::filesystem::path path = CalibrationPath(true);
        if(path.empty())
        {
            Notify(msg_warning, "EpiLightShow can only save its calibration file inside UserData.");
            return false;
        }
        dictionary file;
        if(std::filesystem::exists(path))
        {
            try
            {
                file.load_json(path.string());
            }
            catch(const std::exception &)
            {
                file = dictionary();
            }
        }
        dictionary entry;
        entry["left_eye_first_angle"] = double(orientation.firstAngle[0]);
        entry["left_eye_direction"] = double(orientation.direction[0]);
        entry["right_eye_first_angle"] = double(orientation.firstAngle[1]);
        entry["right_eye_direction"] = double(orientation.direction[1]);
        entry["mouth_swap_rows"] = orientation.swapRows ? 1.0 : 0.0;
        entry["mouth_reverse"] = orientation.reverse ? 1.0 : 0.0;
        file[robot.as_string()] = entry;
        std::ofstream out(path);
        if(!out)
        {
            Notify(msg_warning, "EpiLightShow could not write " + path.string() + ".");
            return false;
        }
        out << file.json() << '\n';
        Notify(msg_print, "EpiLightShow saved the LED orientation of " + robot.as_string() + " in " + path.string() + ".");
        return true;
    }


    void
    StartPattern(int p)
    {
        pattern = p;
        phase = 0;
        patternClock = 0;
        if(colorMode.as_int() == 0)
            hue = Uniform(0.0f, 1.0f);
    }


    void
    HandleCommands()
    {
        std::vector<std::pair<std::string, int>> pending;
        {
            std::lock_guard<std::mutex> lock(commandMutex);
            pending.swap(commands);
        }
        for(const auto & [name, x] : pending)
        {
            const int eye = std::clamp(x, 0, 1);
            if(name == "play")
            {
                checking = false;
                StartPattern(x == 2 ? circle : wave);
                buttonLeft = buttonDuration.as_float();
            }
            else if(name == "stop")
            {
                buttonLeft = 0;
                pattern = none;
            }
            else if(name == "check")
                checking = true;
            else if(name == "rotate")
                orientation.firstAngle[eye] = std::fmod(orientation.firstAngle[eye] + 30.0f, 360.0f);
            else if(name == "flip")
                orientation.direction[eye] = -orientation.direction[eye];
            else if(name == "swap_mouth_rows")
                orientation.swapRows = !orientation.swapRows;
            else if(name == "reverse_mouth")
                orientation.reverse = !orientation.reverse;
            else if(name == "save")
            {
                if(SaveOrientation())
                {
                    calibrated(0) = 1;
                    checking = false;
                }
            }
        }
    }


    // The wave lights one row at a time from the top of the eyes down through the mouth and starts
    // over; the row it has just left glows dimly for a moment.
    void
    DrawWave()
    {
        const float period = waveRows + 1.0f; // one dark row between rounds
        const float p = std::fmod(phase, period);
        auto level = [&](int row)
        {
            const float d = p - row;
            if(d >= 0.0f && d < 1.0f)
                return 1.0f;
            if(d >= 1.0f && d < 1.6f)
                return 0.4f * (1.0f - (d - 1.0f) / 0.6f);
            return 0.0f;
        };
        for(int i = 0; i < eyeLeds; ++i)
        {
            // The row is the number of steps around the ring from the top LED 9: rows 0 to 6.
            const int steps = std::abs(i - 9);
            const int eyeRow = std::min(steps, eyeLeds - steps);
            leftLevel[i] = rightLevel[i] = level(eyeRow);
        }
        for(int i = 0; i < mouthLeds; ++i)
        {
            highLevel[i] = level(7);
            lowLevel[i] = level(8);
        }
    }


    // The circle runs around the eyes in mirrored directions, starting at the top: clockwise as seen
    // from the front in the eye on the viewer's right (the robot's left eye) and counterclockwise in the
    // other. Each half of the mouth is a loop of its upper and lower row that follows the eye on its side.
    void
    DrawCircle()
    {
        const float lap = phase - std::floor(phase);
        const float tailLaps = std::max(tail.as_float(), 0.01f);
        auto trail = [&](float behind)
        {
            behind = behind - std::floor(behind);
            return behind < tailLaps ? 1.0f - behind / tailLaps : 0.0f;
        };
        for(int i = 0; i < eyeLeds; ++i)
        {
            // Position of LED i in laps, clockwise from the top.
            const float fromTop = std::fmod(i / 12.0f - 0.75f + 1.0f, 1.0f);
            leftLevel[i] = trail(lap - fromTop);            // clockwise
            rightLevel[i] = trail(lap - (1.0f - fromTop));  // counterclockwise
        }
        for(int k = 0; k < mouthLeds; ++k)
        {
            // Loop of the viewer's right half (robot's left), clockwise: upper row 4..7, lower row 7..4.
            const int rightHalf = k < 4 ? 4 + k : 7 - (k - 4);
            const bool rightUpper = k < 4;
            // Loop of the viewer's left half, counterclockwise: upper row 3..0, lower row 0..3.
            const int leftHalf = k < 4 ? 3 - k : k - 4;
            const bool leftUpper = k < 4;
            const float position = k / 8.0f;
            const float valueRight = trail(lap - position);
            const float valueLeft = trail(lap - position);
            (rightUpper ? highLevel : lowLevel)[rightHalf] = valueRight;
            (leftUpper ? highLevel : lowLevel)[leftHalf] = valueLeft;
        }
    }


    // Writes colored levels, mixed over the expression, into the widget-layout outputs.
    void
    Compose(const float rgb[3])
    {
        const float b = std::clamp(brightness.as_float(), 0.0f, 1.0f);
        auto blend = [&](const matrix & base, matrix & out, const float * level, int n)
        {
            for(int c = 0; c < 3; ++c)
                for(int i = 0; i < n; ++i)
                {
                    const float patternValue = b * level[i] * rgb[c];
                    out(c, i) = std::clamp((1.0f - mix) * base(c, i) + mix * patternValue, 0.0f, 1.0f);
                }
        };
        blend(eyeLeft, viewLeftEye, leftLevel, eyeLeds);
        blend(eyeRight, viewRightEye, rightLevel, eyeLeds);
        blend(mouthTop, viewMouthHigh, highLevel, mouthLeds);
        blend(mouthBottom, viewMouthLow, lowLevel, mouthLeds);
    }


    // The orientation check: top red, viewer's right green, bottom blue, viewer's left yellow in each
    // eye; the upper mouth row cyan and the lower row magenta, each with its leftmost visible LED white.
    void
    DrawCheck()
    {
        auto eye = [](matrix & out)
        {
            out.set(0.0f);
            const int index[4] = {9, 0, 3, 6};
            const float colors[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}};
            for(int m = 0; m < 4; ++m)
                for(int c = 0; c < 3; ++c)
                    out(c, index[m]) = 0.8f * colors[m][c];
        };
        auto row = [](matrix & out, const float rgb[3])
        {
            for(int i = 0; i < mouthLeds; ++i)
                for(int c = 0; c < 3; ++c)
                    out(c, i) = (i == 1 ? 0.8f : 0.4f * rgb[c]);
        };
        eye(viewLeftEye);
        eye(viewRightEye);
        const float cyan[3] = {0, 1, 1};
        const float magenta[3] = {1, 0, 1};
        row(viewMouthHigh, cyan);
        row(viewMouthLow, magenta);
    }


    // Maps the widget layout to the robot's wiring: physical eye LED p sits at the angle
    // first_angle + direction * 30 p, and the mouth rows may be swapped and reversed.
    void
    MapToRobot()
    {
        auto eye = [&](const matrix & view, matrix & out, int e)
        {
            for(int p = 0; p < eyeLeds; ++p)
            {
                const float angle = orientation.firstAngle[e] + orientation.direction[e] * 30.0f * p;
                const int index = ((static_cast<int>(std::lround(angle / 30.0f)) % eyeLeds) + eyeLeds) % eyeLeds;
                for(int c = 0; c < 3; ++c)
                    out(c, p) = view(c, index);
            }
        };
        eye(viewLeftEye, leftEye, 0);
        eye(viewRightEye, rightEye, 1);
        const matrix & upper = orientation.swapRows ? viewMouthLow : viewMouthHigh;
        const matrix & lower = orientation.swapRows ? viewMouthHigh : viewMouthLow;
        for(int p = 0; p < mouthLeds; ++p)
        {
            const int index = orientation.reverse ? mouthLeds - 1 - p : p;
            for(int c = 0; c < 3; ++c)
            {
                mouthHigh(c, p) = upper(c, index);
                mouthLow(c, p) = lower(c, index);
            }
        }
    }

public:
    void
    Init() override
    {
        Bind(robot, "robot");
        Bind(calibrationFile, "calibration_file");
        Bind(checkAtStart, "check");
        Bind(showPatterns, "show_patterns");
        Bind(patternTime, "pattern_time");
        Bind(buttonDuration, "button_duration");
        Bind(colorMode, "color_mode");
        Bind(brightness, "brightness");
        Bind(waveRate, "wave_rate");
        Bind(circleRate, "circle_rate");
        Bind(tail, "tail");
        Bind(fadeRate, "fade_rate");
        Bind(seed, "seed");

        Bind(eyeLeft, "EYE_LEFT");
        Bind(eyeRight, "EYE_RIGHT");
        Bind(mouthTop, "MOUTH_TOP");
        Bind(mouthBottom, "MOUTH_BOTTOM");
        Bind(run, "RUN");
        Bind(interrupt, "INTERRUPT");
        Bind(arousal, "AROUSAL");
        Bind(color, "COLOR");

        Bind(leftEye, "LEFT_EYE");
        Bind(rightEye, "RIGHT_EYE");
        Bind(mouthHigh, "MOUTH_HIGH");
        Bind(mouthLow, "MOUTH_LOW");
        Bind(viewLeftEye, "VIEW_LEFT_EYE");
        Bind(viewRightEye, "VIEW_RIGHT_EYE");
        Bind(viewMouthHigh, "VIEW_MOUTH_HIGH");
        Bind(viewMouthLow, "VIEW_MOUTH_LOW");
        Bind(patternOutput, "PATTERN");
        Bind(calibrated, "CALIBRATED");
        Bind(orientationOutput, "ORIENTATION");

        auto checkShape = [&](const matrix & m, int n, const char * name)
        {
            if(m.rank() != 2 || m.size_y() != 3 || m.size_x() != n)
                throw exception(std::string("EpiLightShow: ") + name + " must be 3 x " + std::to_string(n) + ".", path_);
        };
        checkShape(eyeLeft, eyeLeds, "EYE_LEFT");
        checkShape(eyeRight, eyeLeds, "EYE_RIGHT");
        checkShape(mouthTop, mouthLeds, "MOUTH_TOP");
        checkShape(mouthBottom, mouthLeds, "MOUTH_BOTTOM");
        for(int i = 0; i < showPatterns.size(); ++i)
            if(static_cast<int>(showPatterns.data()[i]) != wave && static_cast<int>(showPatterns.data()[i]) != circle)
                throw exception("EpiLightShow: show_patterns may only contain 1 (wave) and 2 (circle).", path_);

        if(seed.as_int() < 0)
            randomGenerator.seed(std::random_device{}());
        else
            randomGenerator.seed(static_cast<std::mt19937::result_type>(seed.as_int()));

        const bool known = LoadOrientation();
        calibrated(0) = known ? 1.0f : 0.0f;
        checking = checkAtStart.as_bool() || !known;
        if(!known)
            Notify(msg_warning, "EpiLightShow: the LED orientation of \"" + robot.as_string() +
                                "\" is not in " + calibrationFile.as_string() + "; showing the LED check until it is saved.");
    }


    void
    Command(std::string command_name, dictionary & parameters) override
    {
        const int x = parameters.contains("x") ? parameters["x"].as_int() : 0;
        std::lock_guard<std::mutex> lock(commandMutex);
        commands.emplace_back(command_name, x);
    }


    void
    Tick() override
    {
        const float dt = static_cast<float>(GetTickDuration());
        HandleCommands();

        const bool running = run.connected() && run(0) > 0.5f;
        const bool interrupted = interrupt.connected() && interrupt(0) > 0.5f;

        if(buttonLeft > 0.0f)
            buttonLeft -= dt;
        else if(running && !interrupted && showPatterns.size() > 0)
        {
            // A show cycles through show_patterns, one pattern_time each.
            if(pattern == none)
            {
                showIndex = 0;
                StartPattern(static_cast<int>(showPatterns.data()[0]));
            }
            else if(patternClock >= patternTime.as_float())
            {
                showIndex = (showIndex + 1) % showPatterns.size();
                StartPattern(static_cast<int>(showPatterns.data()[showIndex]));
            }
        }
        else if(pattern != none && (interrupted || !running))
        {
            if(interrupted)
                mix = 0;
            if(mix <= 0.01f)
                pattern = none;
        }

        const bool showing = pattern != none && (buttonLeft > 0.0f || (running && !interrupted));
        const float fade = fadeRate.as_float() <= 0.0f ? 1.0f : 1.0f - std::exp(-fadeRate.as_float());
        mix += fade * ((showing ? 1.0f : 0.0f) - mix);
        if(!showing && mix <= 0.01f)
        {
            mix = 0;
            pattern = none;
        }

        float rgb[3];
        if(color.connected())
            for(int c = 0; c < 3; ++c)
                rgb[c] = std::clamp(color(c), 0.0f, 1.0f);
        else
            HueToColor(colorMode.as_int() == 1 ? 0.6f - 0.55f * Arousal() : hue, rgb);

        std::fill(std::begin(leftLevel), std::end(leftLevel), 0.0f);
        std::fill(std::begin(rightLevel), std::end(rightLevel), 0.0f);
        std::fill(std::begin(highLevel), std::end(highLevel), 0.0f);
        std::fill(std::begin(lowLevel), std::end(lowLevel), 0.0f);
        if(pattern == wave)
            DrawWave();
        else if(pattern == circle)
            DrawCircle();

        if(checking)
            DrawCheck();
        else
            Compose(rgb);
        MapToRobot();

        // Excitement speeds the patterns up, from 0.7 to 1.5 times; the first frame shows the start.
        const float speed = 0.7f + 0.8f * Arousal();
        if(pattern != none)
        {
            patternClock += dt;
            phase += dt * speed * (pattern == wave ? waveRate.as_float() : circleRate.as_float());
        }

        patternOutput(0) = checking ? static_cast<float>(check) : static_cast<float>(pattern);
        orientationOutput(0) = orientation.firstAngle[0];
        orientationOutput(1) = orientation.direction[0];
        orientationOutput(2) = orientation.firstAngle[1];
        orientationOutput(3) = orientation.direction[1];
        orientationOutput(4) = orientation.swapRows ? 1.0f : 0.0f;
        orientationOutput(5) = orientation.reverse ? 1.0f : 0.0f;
    }
};

INSTALL_CLASS(EpiLightShow)
