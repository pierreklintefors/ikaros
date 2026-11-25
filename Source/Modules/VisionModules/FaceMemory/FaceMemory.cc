#include "ikaros.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace ikaros;

class FaceMemory : public Module
{
    static constexpr float degreesPerRadian = 57.29577951308232f;
    static constexpr float continuityTime = 1.0f; // seconds a face counts as still in view for matching
    static constexpr float closerRatio = 2.0f;    // a face this many times wider is someone much closer
    static constexpr float returnInterval = 1.0f; // seconds between requests to turn back to a face
    static constexpr float callTimeout = 1.0f;    // seconds a call waits for a measured frame with a face near it

    struct Detection
    {
        int box = 0;
        float x = 0;
        float y = 0;
        float elevation = 0;
        float azimuth = 0;
        bool matched = false;
    };

    struct Face
    {
        bool used = false;
        int id = 0;
        float elevation = 0;
        float azimuth = 0;
        float width = 0;
        float x = 0;            // image center in the last measured frame
        float y = 0;
        bool visible = false;   // detected in the last measured frame
        int sightings = 0;      // measured frames in which the face was detected
        float firstSeen = 0;
        float lastSeen = 0;
        float lastLooked = -1;  // last time in a look that has been judged; -1 when never looked at
    };

    parameter horizontalFov;
    parameter verticalFov;
    parameter xSign;
    parameter ySign;
    parameter minWidth;
    parameter minConfidence;
    parameter capacity;
    parameter matchRadius;
    parameter maxCameraSpeed;
    parameter stillTime;
    parameter forgetTime;
    parameter minSeen;
    parameter minDwell;
    parameter maxDwell;
    parameter regreetTime;
    parameter returnTime;
    parameter callRadius;
    parameter seed;

    matrix boxes;
    matrix camera;
    matrix target;
    matrix busy;
    matrix call;

    matrix preferred;
    matrix newFace;
    matrix dwell;
    matrix count;
    matrix faces;

    std::vector<Face> memory;
    std::vector<float> weights;        // work buffer for choosing the next face
    std::vector<float> previousBoxes;  // detector output of the previous tick, to recognize repeated frames
    std::vector<Detection> detections; // accepted faces of the present frame
    std::mt19937 randomGenerator;
    float now = 0;
    int nextId = 1;
    int followedId = 0;                // face looked at last, 0 for none
    int intendedId = 0;                // face the robot means to look at, 0 for none
    float slipTime = 0;                // seconds the selector has followed another face than the intended one
    float lastReturn = -1000;          // when the robot last asked to turn back to the intended face
    bool callPending = false;          // a call waits for a measured frame
    float callElevation = 0;
    float callAzimuth = 0;
    float callTime = 0;
    float previousCallFlag = 0;
    float lookStart = 0;               // when the present look began
    bool judged = true;                // whether the present look has been judged new or not
    float dwellTime = 0;
    float dwellLimit = 0;
    float previousCamera[2] = {0, 0};
    float stillFor = 0;                // seconds the camera has been slower than max_camera_speed

    float
    Uniform(float low, float high)
    {
        if(high <= low)
            return low;
        return std::uniform_real_distribution<float>(low, high)(randomGenerator);
    }


    int
    BoxCount() const
    {
        if(boxes.empty())
            return 0;
        if(boxes.rank() == 2)
            return boxes.cols() >= 5 ? boxes.rows() : 0;
        if(boxes.rank() == 1)
            return boxes.size() >= 5 ? 1 : 0;
        return 0;
    }


    float
    Box(int index, int field) const
    {
        if(boxes.rank() == 2)
            return boxes(index, field);
        return boxes(field);
    }


    bool
    Accepted(int index) const
    {
        const float width = Box(index, 2);
        const float confidence = Box(index, 4);
        return std::isfinite(width) && std::isfinite(confidence) &&
               width >= minWidth.as_float() && width > 0.0f &&
               confidence >= minConfidence.as_float();
    }


    // The same detector frame gives exactly the same boxes, so identical output means no new frame.
    bool
    RepeatedDetection()
    {
        const int n = BoxCount();
        const std::size_t valueCount = static_cast<std::size_t>(n) * 5;
        bool repeated = n > 0 && previousBoxes.size() == valueCount;

        if(previousBoxes.size() != valueCount)
            previousBoxes.resize(valueCount);

        for(int i = 0; i < n; ++i)
            for(int field = 0; field < 5; ++field)
            {
                float & previous = previousBoxes[static_cast<std::size_t>(i) * 5 + field];
                const float current = Box(i, field);
                if(previous != current)
                    repeated = false;
                previous = current;
            }
        return repeated;
    }


    float
    ImageAngle(float coordinate, float fieldOfView, float sign) const
    {
        const float halfField = 0.5f * fieldOfView / degreesPerRadian;
        return sign * std::atan(coordinate * std::tan(halfField)) * degreesPerRadian;
    }


    void
    Direction(float x, float y, float & elevation, float & azimuth) const
    {
        elevation = camera(0) + ImageAngle(y, verticalFov.as_float(), ySign.as_float());
        azimuth = camera(1) + ImageAngle(x, horizontalFov.as_float(), xSign.as_float());
    }


    static float
    Distance(const Face & face, float elevation, float azimuth)
    {
        return std::hypot(face.elevation - elevation, face.azimuth - azimuth);
    }


    // Whether a face last looked at lastLooked counts as new at time t.
    bool
    NotLookedAt(float lastLooked, float t) const
    {
        return lastLooked < 0.0f || t - lastLooked > regreetTime.as_float();
    }


    // Whether another face looked at within regreet_time is remembered within half match_radius; a new
    // entry that close is the same person stored twice from a misjudged direction, not a new person.
    bool
    KnownNearby(int index, float t) const
    {
        const Face & face = memory[index];
        for(int i = 0; i < static_cast<int>(memory.size()); ++i)
        {
            const Face & other = memory[i];
            if(i != index && other.used && !NotLookedAt(other.lastLooked, t) &&
               Distance(other, face.elevation, face.azimuth) <= 0.5f * matchRadius.as_float())
                return true;
        }
        return false;
    }


    bool
    IsNew(int index) const
    {
        return NotLookedAt(memory[index].lastLooked, now) && !KnownNearby(index, now);
    }


    // A free slot, or the face seen longest ago that is neither in view nor followed.
    int
    Slot() const
    {
        int oldest = -1;
        for(int i = 0; i < static_cast<int>(memory.size()); ++i)
        {
            const Face & face = memory[i];
            if(!face.used)
                return i;
            if(face.visible || face.id == followedId)
                continue;
            if(oldest < 0 || face.lastSeen < memory[oldest].lastSeen)
                oldest = i;
        }
        return oldest;
    }


    void
    BoxCenter(int index, float & x, float & y) const
    {
        x = Box(index, 0) + 0.5f * Box(index, 2);
        y = Box(index, 1) + 0.5f * Box(index, 3);
    }


    // A known face moves halfway toward each measurement, so a single frame from a misjudged camera
    // direction cannot carry it out of reach of its next correct measurement.
    void
    Update(Face & face, const Detection & detection)
    {
        const float blend = face.sightings > 0 ? 0.5f : 1.0f;
        face.elevation += blend * (detection.elevation - face.elevation);
        face.azimuth += blend * (detection.azimuth - face.azimuth);
        face.width = Box(detection.box, 2);
        face.x = detection.x;
        face.y = detection.y;
        face.visible = true;
        face.lastSeen = now;
        ++face.sightings;
    }


    // Gives each detected face to the remembered face nearest to it, closest pairs first. Faces seen
    // within continuityTime are matched first, so a person in view keeps the same memory even when an
    // older entry lies nearer; the remaining detections are matched to faces out of view, and the rest
    // are remembered as new faces. A face seen in only one measured frame and missing from the next is
    // dropped, because a frame taken while the camera started to move places a face in a misjudged
    // direction; the person then matches their real entry again once the camera has settled.
    void
    Measure()
    {
        for(Face & face : memory)
        {
            if(face.used && face.sightings < 2 && !face.visible && face.id != followedId)
                face.used = false;
            face.visible = false;
        }

        detections.clear();
        const int n = BoxCount();
        for(int i = 0; i < n; ++i)
        {
            if(!Accepted(i))
                continue;
            Detection detection;
            detection.box = i;
            BoxCenter(i, detection.x, detection.y);
            Direction(detection.x, detection.y, detection.elevation, detection.azimuth);
            detections.push_back(detection);
        }

        for(int pass = 0; pass < 2; ++pass)
            while(true)
            {
                int bestDetection = -1;
                int bestFace = -1;
                float bestDistance = matchRadius.as_float();
                for(int d = 0; d < static_cast<int>(detections.size()); ++d)
                {
                    if(detections[d].matched)
                        continue;
                    for(int j = 0; j < static_cast<int>(memory.size()); ++j)
                    {
                        const Face & face = memory[j];
                        if(!face.used || face.visible || (pass == 0 && now - face.lastSeen > continuityTime))
                            continue;
                        const float distance = Distance(face, detections[d].elevation, detections[d].azimuth);
                        if(distance <= bestDistance)
                        {
                            bestDistance = distance;
                            bestDetection = d;
                            bestFace = j;
                        }
                    }
                }
                if(bestDetection < 0)
                    break;
                Update(memory[bestFace], detections[bestDetection]);
                detections[bestDetection].matched = true;
            }

        for(const Detection & detection : detections)
        {
            if(detection.matched)
                continue;
            const int slot = Slot();
            if(slot < 0)
                break;
            memory[slot] = Face();
            memory[slot].used = true;
            memory[slot].id = nextId++;
            memory[slot].firstSeen = now;
            Update(memory[slot], detection);
        }
    }


    int
    IndexOf(int id) const
    {
        for(int i = 0; i < static_cast<int>(memory.size()); ++i)
            if(memory[i].used && memory[i].id == id)
                return i;
        return -1;
    }


    // The face in view nearest the direction of the selector's target. Only faces detected in the last
    // measured frame count, each of which is one person, so an old entry near the person cannot take
    // over; without one, the face followed before is kept.
    int
    Followed() const
    {
        if(target(2) <= 0.0f)
            return -1;
        float elevation = 0;
        float azimuth = 0;
        Direction(target(0), target(1), elevation, azimuth);

        int nearest = -1;
        float nearestDistance = matchRadius.as_float();
        for(int i = 0; i < static_cast<int>(memory.size()); ++i)
        {
            if(!memory[i].used || !memory[i].visible)
                continue;
            const float distance = Distance(memory[i], elevation, azimuth);
            if(distance <= nearestDistance)
            {
                nearestDistance = distance;
                nearest = i;
            }
        }
        return nearest >= 0 ? nearest : IndexOf(followedId);
    }


    // A face in view that has not been looked at is taken at once, the largest first. Otherwise, once
    // the dwell time is over, a face is drawn with a weight that grows with the time since it was
    // looked at, so the robot looks around the group instead of going back and forth.
    int
    ChooseNext(int followed)
    {
        int newest = -1;
        int candidates = 0;
        float total = 0;
        for(int i = 0; i < static_cast<int>(memory.size()); ++i)
        {
            const Face & face = memory[i];
            weights[i] = 0;
            if(!face.used || !face.visible || i == followed || now - face.firstSeen < minSeen.as_float())
                continue;
            if(IsNew(i))
            {
                if(newest < 0 || face.width > memory[newest].width)
                    newest = i;
                continue;
            }
            ++candidates;
            weights[i] = std::max(now - face.lastLooked, 0.001f);
            total += weights[i];
        }

        if(newest >= 0)
            return newest;
        if(candidates == 0 || dwellTime < dwellLimit)
            return -1;

        float pick = Uniform(0.0f, total);
        int last = -1;
        for(int i = 0; i < static_cast<int>(memory.size()); ++i)
        {
            if(weights[i] <= 0.0f)
                continue;
            if(pick < weights[i])
                return i;
            pick -= weights[i];
            last = i;
        }
        return last;
    }


    // Whether the selector follows another face than the one FaceMemory chose, usually because it lost
    // that face for a moment and took the largest one in view. Such a switch is undone for return_time
    // while the dwell on the intended face is not over, unless the other face is much closer.
    bool
    Slipped(int followed) const
    {
        const int intended = IndexOf(intendedId);
        if(intended < 0 || intended == followed || slipTime >= returnTime.as_float())
            return false;
        const Face & before = memory[intended];
        return dwellTime < dwellLimit && now - before.lastSeen <= returnTime.as_float() &&
               memory[followed].width < closerRatio * before.width;
    }


    // A rising CALL flag, such as a wave, is stored as a direction; on the next measured frame the robot
    // turns to the face in view nearest to it, or stays with that face if it is the one looked at. The
    // face counts as looked at, so the call is answered by the caller's own reaction, not by NEW_FACE.
    void
    Call(int followed, bool measured)
    {
        if(!call.connected())
            return;
        const float flag = call(2);
        if(flag > 0.5f && previousCallFlag <= 0.5f)
        {
            Direction(call(0), call(1), callElevation, callAzimuth);
            callPending = true;
            callTime = now;
        }
        previousCallFlag = flag;

        if(!callPending || !measured)
            return;
        if(now - callTime > callTimeout)
        {
            callPending = false;
            return;
        }

        int nearest = -1;
        float nearestDistance = callRadius.as_float();
        for(int i = 0; i < static_cast<int>(memory.size()); ++i)
        {
            if(!memory[i].used || !memory[i].visible)
                continue;
            const float distance = Distance(memory[i], callElevation, callAzimuth);
            if(distance <= nearestDistance)
            {
                nearestDistance = distance;
                nearest = i;
            }
        }
        if(nearest < 0)
            return;

        callPending = false;
        Face & caller = memory[nearest];
        caller.lastLooked = now;
        intendedId = caller.id;
        if(nearest != followed)
        {
            preferred(0) = caller.x;
            preferred(1) = caller.y;
            preferred(2) = 1;
            lastReturn = now;
        }
        StartDwell();
    }


    void
    StartDwell()
    {
        dwellTime = 0;
        dwellLimit = Uniform(minDwell.as_float(), maxDwell.as_float());
    }


    void
    WriteFaces()
    {
        faces.reset();
        int row = 0;
        for(const Face & face : memory)
        {
            if(!face.used)
                continue;
            faces(row, 0) = face.elevation;
            faces(row, 1) = face.azimuth;
            faces(row, 2) = face.width;
            faces(row, 3) = now - face.lastSeen;
            faces(row, 4) = face.lastLooked < 0.0f ? -1.0f : now - face.lastLooked;
            ++row;
        }
        count(0) = static_cast<float>(row);
    }

public:
    void
    Init() override
    {
        Bind(horizontalFov, "horizontal_fov");
        Bind(verticalFov, "vertical_fov");
        Bind(xSign, "x_sign");
        Bind(ySign, "y_sign");
        Bind(minWidth, "min_width");
        Bind(minConfidence, "min_confidence");
        Bind(capacity, "capacity");
        Bind(matchRadius, "match_radius");
        Bind(maxCameraSpeed, "max_camera_speed");
        Bind(stillTime, "still_time");
        Bind(forgetTime, "forget_time");
        Bind(minSeen, "min_seen");
        Bind(minDwell, "min_dwell");
        Bind(maxDwell, "max_dwell");
        Bind(regreetTime, "regreet_time");
        Bind(returnTime, "return_time");
        Bind(callRadius, "call_radius");
        Bind(seed, "seed");

        Bind(boxes, "BOXES");
        Bind(camera, "CAMERA");
        Bind(target, "TARGET");
        Bind(busy, "BUSY");
        Bind(call, "CALL");

        Bind(preferred, "PREFERRED");
        Bind(newFace, "NEW_FACE");
        Bind(dwell, "DWELL");
        Bind(count, "COUNT");
        Bind(faces, "FACES");

        if(target.size() < 3)
            throw exception("FaceMemory: TARGET must contain x, y, and width.", path_);
        if(faces.rank() != 2 || faces.rows() != capacity.as_int() || faces.cols() != 5)
            throw exception("FaceMemory: FACES must have capacity rows of 5 values.", path_);

        if(seed.as_int() < 0)
            randomGenerator.seed(std::random_device{}());
        else
            randomGenerator.seed(static_cast<std::mt19937::result_type>(seed.as_int()));

        memory.resize(static_cast<std::size_t>(capacity.as_int()));
        weights.resize(memory.size());
        previousBoxes.reserve(16 * 5);
        detections.reserve(32);
        StartDwell();
    }


    void
    Tick() override
    {
        const float dt = static_cast<float>(GetTickDuration());
        now += dt;
        preferred.reset();
        newFace.reset();

        const float cameraSpeed = dt > 0.0f ?
            std::hypot(camera(0) - previousCamera[0], camera(1) - previousCamera[1]) / dt : 0.0f;
        previousCamera[0] = camera(0);
        previousCamera[1] = camera(1);
        stillFor = cameraSpeed <= maxCameraSpeed.as_float() ? stillFor + dt : 0.0f;
        // The camera direction and the image are not taken at exactly the same moment, so a short
        // time after each movement must pass before they line up.
        const bool still = stillFor >= stillTime.as_float() - 1e-5f;

        const bool measured = !RepeatedDetection() && still;
        if(measured)
            Measure();

        for(Face & face : memory)
            if(face.used && now - face.lastSeen > forgetTime.as_float())
                face.used = false;

        // When the selector has let go of its target, finding a face again counts as a new look.
        if(target(2) <= 0.0f)
            followedId = 0;
        // While the camera moves, the target's direction is uncertain, so the followed face is kept.
        int followed = still ? Followed() : IndexOf(followedId);
        Call(followed, measured);
        if(followed >= 0 && Slipped(followed))
        {
            // The look at the intended face goes on; the other face does not count as looked at.
            slipTime += dt;
            const Face & intended = memory[IndexOf(intendedId)];
            if(measured && intended.visible && now - lastReturn >= returnInterval - 1e-5f)
            {
                preferred(0) = intended.x;
                preferred(1) = intended.y;
                preferred(2) = 1;
                lastReturn = now;
            }
            followed = -1;
        }
        else
        {
            slipTime = 0;
            if(followed >= 0)
                intendedId = memory[followed].id;
        }

        const bool isBusy = busy.connected() && busy(0) > 0.5f;
        if(followed >= 0)
        {
            Face & face = memory[followed];
            if(face.id != followedId)
            {
                followedId = face.id;
                lookStart = now;
                judged = false;
                StartDwell();
            }

            // Whether the face is new is judged once the look has lasted min_seen and the camera has
            // settled, because the first frames after a turn can place the face in a misjudged direction.
            // Until then the look does not count, so a look that is cut short leaves the face unmet.
            if(!judged && measured && face.visible && now - lookStart >= minSeen.as_float() - 1e-5f)
            {
                judged = true;
                if(IsNew(followed))
                    newFace(0) = 1;
            }
            if(judged)
                face.lastLooked = now;
            if(!isBusy)
                dwellTime += dt;

            if(measured && !isBusy && dwellTime >= minDwell.as_float())
            {
                const int next = ChooseNext(followed);
                if(next >= 0)
                {
                    preferred(0) = memory[next].x;
                    preferred(1) = memory[next].y;
                    preferred(2) = 1;
                    intendedId = memory[next].id;
                    lastReturn = now; // the selector needs a new frame to turn; do not ask again before
                    StartDwell();
                }
            }
        }

        dwell(0) = dwellTime;
        WriteFaces();
    }
};

INSTALL_CLASS(FaceMemory)
