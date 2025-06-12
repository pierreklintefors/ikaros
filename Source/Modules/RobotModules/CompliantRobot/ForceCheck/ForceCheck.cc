#include "ikaros.h"
#include <chrono>

using namespace ikaros;

// Base class for control modes
class ControlMode {
protected:
    matrix& force_output;
    matrix& goal_position_out;
    matrix& led_intensity;
    matrix& led_color_eyes;
    matrix& led_color_mouth;
    
public:
    ControlMode(matrix& force_out, matrix& goal_out, matrix& led_int, 
                matrix& led_eyes, matrix& led_mouth)
        : force_output(force_out), goal_position_out(goal_out), 
          led_intensity(led_int), led_color_eyes(led_eyes), led_color_mouth(led_mouth) {}
    
    virtual ~ControlMode() = default;
    virtual void HandleDeviation(matrix& deviation, matrix& present_position,
                                matrix& goal_position_in, matrix& start_position,
                                matrix& allowed_deviance, matrix& started_transition,
                                matrix& number_deviations_per_time_window, matrix& torque,
                                double time_window, double force_change_rate) = 0;
    virtual void SetLEDColor(double deviance_ratio) = 0;
    virtual const char* GetModeName() const = 0;
};

// Normal pose tracking mode
class NormalMode : public ControlMode {
public:
    NormalMode(matrix& force_out, matrix& goal_out, matrix& led_int, 
               matrix& led_eyes, matrix& led_mouth)
        : ControlMode(force_out, goal_out, led_int, led_eyes, led_mouth) {}
    
    void HandleDeviation(matrix& deviation, matrix& present_position,
                        matrix& goal_position_in, matrix& start_position,
                        matrix& allowed_deviance, matrix& started_transition,
                        matrix& number_deviations_per_time_window, matrix& torque,
                        double time_window, double force_change_rate) override {

        
        force_output.set(clip(force_output[0] + force_change_rate, 0, 100));
    }
    
    void SetLEDColor(double deviance_ratio) override {
        matrix color(3);
        if (deviance_ratio < 0.3) {
            // White for normal operation
            color.set(1.0);
        } else {
            // Orange for medium deviances
            color(0) = 1.0; // Red
            color(1) = 0.5; // Green
            color(2) = 0.0; // Blue
        }
        
        // Apply to all LEDs
        for (int i = 0; i < 12; i++) {
            led_color_eyes(0, i) = color(0);
            led_color_eyes(1, i) = color(1);
            led_color_eyes(2, i) = color(2);
        }
        for (int i = 0; i < 8; i++) {
            led_color_mouth(0, i) = color(0);
            led_color_mouth(1, i) = color(1);
            led_color_mouth(2, i) = color(2);
        }
    }
    
    const char* GetModeName() const override { return "Normal"; }
};

// Avoidant mode - pulls away from high deviation
class AvoidantMode : public ControlMode {
private:
    bool obstacle_detected =false;
    std::chrono::steady_clock::time_point last_avoidance_time;
public:
    // Constructor initializes the base class with references to output matrices
    AvoidantMode(matrix& force_out, matrix& goal_out, matrix& led_int, 
                 matrix& led_eyes, matrix& led_mouth)
        : ControlMode(force_out, goal_out, led_int, led_eyes, led_mouth) {}
    
    void HandleDeviation( matrix& deviation,  matrix& present_position,
                        matrix& goal_position_in,  matrix& start_position,
                        matrix& allowed_deviance,  matrix& started_transition,
                        matrix& number_deviations_per_time_window, matrix& torque,  
                        double time_window, double force_change_rate) override {
        if (present_position.size() == 0 || goal_position_in.size() == 0 || goal_position_out.size() == 0) return;

        // Default to passing through the goal_position_in from GoalSetter.
        // This will be overridden if any specific motor needs to avoid.
        goal_position_out.copy(goal_position_in);

        bool an_obstacle_requires_active_avoidance_this_tick = false;

        // Check each motor for high deviation to determine if avoidance is needed *now*.
        for (int i = 0; i < deviation.size() && i < started_transition.size() && i < goal_position_out.size(); i++) {
            if (started_transition[i] == 1 && abs(deviation[i]) > allowed_deviance[i]) {
                // This motor requires avoidance. Calculate and set its part of goal_position_out.
                double movement_direction = ((double)present_position[i] > (double)start_position[i]) ? -5.0 : 5.0;
                goal_position_out(i) = present_position(i) + movement_direction;
                an_obstacle_requires_active_avoidance_this_tick = true;
            }
        }

        if (an_obstacle_requires_active_avoidance_this_tick) {
            // If any motor is actively avoiding, set/refresh the general obstacle detection state.
            obstacle_detected = true;
            last_avoidance_time = std::chrono::steady_clock::now();
        } else {
            // No motor requires active avoidance *this tick*.
            // Check if a general obstacle state (from a previous tick) should time out.
            if (obstacle_detected) { // obstacle_detected was true from a previous tick.
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_avoidance_time).count() > 1000) { // 1 second timeout
                    obstacle_detected = false; // Timeout the general obstacle state.
                }
                // If obstacle_detected is still true here (timer not expired),
                // goal_position_out is already goal_position_in (pass-through),
                // as no motor needed specific avoidance *this tick*.
            }
            // If obstacle_detected was already false, it remains false.
            // goal_position_out is already goal_position_in (pass-through).
        }
        
    }
    
    void SetLEDColor(double deviance_ratio) override {
        matrix color(3);
        if (deviance_ratio > 0.8) {
            // Red for high deviation
            color(0) = 1.0; color(1) = 0.0; color(2) = 0.0;
        } else {
            // Yellow for avoidance mode
            color(0) = 1.0; color(1) = 1.0; color(2) = 0.0;
        }
        
        // Apply to all LEDs
        for (int i = 0; i < 12; i++) {
            led_color_eyes(0, i) = color(0);
            led_color_eyes(1, i) = color(1);
            led_color_eyes(2, i) = color(2);
        }
        for (int i = 0; i < 8; i++) {
            led_color_mouth(0, i) = color(0);
            led_color_mouth(1, i) = color(1);
            led_color_mouth(2, i) = color(2);
        }
    }
    
    const char* GetModeName() const override { return "Avoidant"; }
};

// Compliant mode - follows external force
class CompliantMode : public ControlMode {
private:
    bool torque_disabled;
    std::chrono::steady_clock::time_point torque_disable_time;
    
public:
    CompliantMode(matrix& force_out, matrix& goal_out, matrix& led_int, 
                  matrix& led_eyes, matrix& led_mouth)
        : ControlMode(force_out, goal_out, led_int, led_eyes, led_mouth), 
          torque_disabled(false) {}
    
    void HandleDeviation(matrix& deviation, matrix& present_position,
                        matrix& goal_position_in, matrix& start_position,
                        matrix& allowed_deviance, matrix& started_transition,
                        matrix& number_deviations_per_time_window, matrix& torque,
                        double time_window, double force_change_rate) override {
        if (present_position.size() == 0 || goal_position_in.size() == 0) return;
        
        // Check for prolonged high deviance in motors that have started transition
        bool high_deviance_detected = false;
        for (int i = 0; i < number_deviations_per_time_window.size() && i < started_transition.size(); i++) {
            if (started_transition[i] == 1 && number_deviations_per_time_window[i] > time_window * 0.5) {
                high_deviance_detected = true;
                break;
            }
        }
        
        if (high_deviance_detected && !torque_disabled) {
            torque_disabled = true;
            torque_disable_time = std::chrono::steady_clock::now();
            torque.set(0); // Disable torque
        } else if (!high_deviance_detected && torque_disabled) {
            // Check if position has stabilized to re-enable torque
            matrix last_position;
            int position_margin = 3; // Margin for position stabilityß
            static auto last_check_time = std::chrono::steady_clock::now();
            static int stable_count = 0;

            last_position= present_position.last(); // Store last position
            
            for (int i = 0; i < present_position.size(); i++) {
                if (abs(present_position(i) - last_position(i)) < position_margin) {
                    stable_count++;
                } else {
                    stable_count = 0; // Reset if any motor deviates
                }  
            }
            std::cout << "ForceCheck: Compliance motor, stability count: " << std::to_string(stable_count) << std::endl;

            if (stable_count >= 150) { // 150 counts for any motor (arbitary number)
                torque_disabled = false;
                stable_count = 0; // Reset for next check
            }
        }
        
        if (!torque_disabled) {
            // Follow the external force by updating goal to current position
            goal_position_out.copy(present_position);
            torque.set(1); // Enable torque
        } else {
            goal_position_out.copy(present_position); // Follow current position
        }
    }
    
    void SetLEDColor(double deviance_ratio) override {
        // Green for compliant mode
        matrix color(3);
        if (torque_disabled) {
            // If torque is disabled, use intense green
            color(0) = 0.0; // Red
            color(1) = 1.0; // Green
            color(2) = 0.0; // Blue
        } else {
            // Normal compliant mode color
            color(0) = 0.2; // Red
            color(1) = 0.5; // Green
            color(2) = 0.5; // Blue
        }
        
        // Apply to all LEDs
        for (int i = 0; i < 12; i++) {
            led_color_eyes(0, i) = color(0);
            led_color_eyes(1, i) = color(1);
            led_color_eyes(2, i) = color(2);
        }
        for (int i = 0; i < 8; i++) {
            led_color_mouth(0, i) = color(0);
            led_color_mouth(1, i) = color(1);
            led_color_mouth(2, i) = color(2);
        }
    }
    
    const char* GetModeName() const override { return "Compliant"; }
};

// Controller class to manage modes
class ModeController {
private:
    std::unique_ptr<ControlMode> current_mode;
    int mode_index;
    
public:
    ModeController(matrix& force_out, matrix& goal_out, matrix& led_int, 
                   matrix& led_eyes, matrix& led_mouth) : mode_index(0) {
        // Start with normal mode
        current_mode = std::make_unique<NormalMode>(force_out, goal_out, led_int, led_eyes, led_mouth);
    }
    
    void SwitchMode(int new_mode, matrix& force_out, matrix& goal_out, matrix& led_int, 
                    matrix& led_eyes, matrix& led_mouth) {
        mode_index = new_mode % 3; // 0=Normal, 1=Avoidant, 2=Compliant
        
        switch (mode_index) {
            case 0:
                current_mode = std::make_unique<NormalMode>(force_out, goal_out, led_int, led_eyes, led_mouth);
                break;
            case 1:
                current_mode = std::make_unique<AvoidantMode>(force_out, goal_out, led_int, led_eyes, led_mouth);
                break;
            case 2:
                current_mode = std::make_unique<CompliantMode>(force_out, goal_out, led_int, led_eyes, led_mouth);
                break;
        }
    }
    
    void HandleDeviation(matrix& deviation, matrix& present_position,
                        matrix& goal_position_in, matrix& start_position,
                        matrix& allowed_deviance, matrix& started_transition,
                        matrix& number_deviations_per_time_window, matrix& torque,
                        double time_window, double force_change_rate) {
        current_mode->HandleDeviation(deviation, present_position, goal_position_in, 
                                    start_position, allowed_deviance, started_transition,
                                    number_deviations_per_time_window, torque, time_window, force_change_rate);
    }
    
    void SetLEDColor(double deviance_ratio) {
        current_mode->SetLEDColor(deviance_ratio);
    }
    
    const char* GetCurrentModeName() const {
        return current_mode->GetModeName();
    }
    
    int GetModeIndex() const { return mode_index; }
};

class ForceCheck: public Module
{

    //parameters
    
    parameter force_change_rate;
    parameter time_window;
    parameter control_mode; // Add this parameter for mode switching
    
    //inputs
    matrix present_current;
    matrix current_limit;
    matrix present_position;// assumes degrees
    matrix goal_position_in;
    matrix start_position;
    matrix allowed_deviance;

    //outputs
    matrix goal_position_out;
    matrix current_prediction;
    matrix force_output;
    matrix deviation;
    matrix led_intensity;
    matrix led_color_eyes;
    matrix led_color_mouth;
    matrix torque;
    
    //Internal
    matrix previous_position;
    matrix red_color_RGB;
    matrix started_transition;
    int tickCount;
    int current_increment;
    int current_value;
    int position_margin;
    int current_margin;
    int minimum_current;
    matrix number_deviations_per_time_window;
   
    bool firstTick;
    bool obstacle;
    double obstacle_time;
    bool goal_reached;
    double start_time;
    int goal_time_out;
    bool force_output_tapped;
    std::chrono::steady_clock::time_point force_output_tapped_time_point;
    std::chrono::steady_clock::time_point time_window_start;

    // Add mode controller
    std::unique_ptr<ModeController> mode_controller;

        double
        Tapering(double error, double threshold)
    {
        if (abs(error) < threshold)
            return sin(M_PI_2)*(abs(error)/threshold);
        else
            return 1;
    }
    
    // Calculate RGB color based on deviance ratio (0.0 to 1.0+)
    // 0.0 = white, 0.33 = yellow, 0.66 = orange, 1.0+ = red
    matrix CalculateDevianceColor(double deviance_ratio)
    {
        matrix color(3); // RGB
        Debug("ForceCheck: Deviance ratio: " + std::to_string(deviance_ratio));
        deviance_ratio = clip(deviance_ratio, 0.0, 1.0); // Ensure ratio is between 0 and 1
        
        // Define color transition points (ratio, R, G, B)
        const double color_points[][4] = {
            {0.0,   1.0, 1.0, 1.0},  // White
            {0.20,  0.9, 1.0, 0.9},  // Very Light Cyan
            {0.40,  0.8, 1.0, 0.8},  // Light Cyan
            {0.50,  0.9, 1.0, 0.6},  // Cyan-Yellow Transition
            {0.55,  1.0, 1.0, 0.5},  // Light Yellow-Cyan
            {0.60,  1.0, 1.0, 0.4},  // Light Yellow
            {0.62,  1.0, 1.0, 0.3},  // Yellow-Cyan
            {0.64,  1.0, 1.0, 0.2},  // Yellow Transition
            {0.66,  1.0, 1.0, 0.1},  // Bright Yellow
            {0.68,  1.0, 1.0, 0.0},  // Pure Yellow
            {0.70,  1.0, 0.8, 0.0},  // Yellow-Orange
            {0.72,  1.0, 0.6, 0.0},  // Light Orange
            {0.74,  1.0, 0.5, 0.0},  // Orange
            {0.76,  1.0, 0.4, 0.0},  // Dark Orange
            {0.78,  1.0, 0.3, 0.0},  // Deep Orange
            {0.80,  1.0, 0.2, 0.0},  // Very Deep Orange
            {0.85,  1.0, 0.1, 0.0},  // Orange-Red
            {0.90,  1.0, 0.05, 0.0}, // Red-Orange
            {0.95,  1.0, 0.02, 0.0}, // Almost Red
            {1.0,   1.0, 0.0, 0.0}   // Pure Red
        };
        const int num_points = sizeof(color_points) / sizeof(color_points[0]);
        
        // Find the appropriate color segment
        for (int i = 0; i < num_points - 1; i++) {
            if (deviance_ratio <= color_points[i + 1][0]) {
                // Calculate interpolation factor
                double t = (deviance_ratio - color_points[i][0]) / 
                          (color_points[i + 1][0] - color_points[i][0]);
                
                // Interpolate each color component
                for (int c = 0; c < 3; c++) {
                    color(c) = color_points[i][c + 1] + 
                             t * (color_points[i + 1][c + 1] - color_points[i][c + 1]);
                }
                break;
            }
        }
        
        return color;
    }

    matrix PredictionDeviance(matrix current_prediction, matrix current_output){
        matrix deviance(current_prediction.size());
        for (int i = 0; i < current_prediction.size(); i++) {
            int prediction = current_prediction[i];
            int output = current_output[i];
            deviance[i] = abs(prediction) - abs(output);
        }
        return deviance;
    }

    
    

    bool GoalReached(matrix present_position, matrix goal_position, int margin)
    {
    
        for (int i = 0; i < present_position.size(); i++) {
            int position = present_position[i];
            int goal = goal_position[i];

            if (abs(position - goal) > margin)
                return false;
        }
        return true;
    }

    // Returns a matrix where each element indicates if the corresponding motor
    // has started its transition from its start_position.
    // 1 for started, 0 for not started.
    matrix StartedTransition(matrix& present_position, matrix& start_position, int margin)
    {
        if (present_position.size() == 0 || present_position.size() != start_position.size())
        {
            Warning("ForceCheck: StartedTransition - Input matrix size mismatch or empty. Returning empty matrix.");
            return matrix(0); 
        }

        matrix transition_status(present_position.size());
        for (int i = 0; i < start_position.size(); i++){
            if (abs(start_position(i) - present_position(i)) < margin || abs(present_current(i))<5)
                transition_status(i) = 0.0; // Not started
            else
                transition_status(i) = 1.0; // Started
        }
        
        return transition_status;
    }

    void Init()
    {
        Bind(present_current, "PresentCurrent");
        Bind(force_output, "ForceOutput");
        Bind(present_position, "PresentPosition");
        Bind(goal_position_in, "GoalPositionIn");
        Bind(goal_position_out, "GoalPositionOut");
        Bind(current_prediction, "CurrentPrediction");
        Bind(deviation, "Deviation");
        Bind(start_position, "StartPosition");
        Bind(allowed_deviance, "AllowedDeviance");
        Bind(force_change_rate, "ForceChangeRate");
        Bind(led_intensity, "LedIntensity");
        Bind(led_color_eyes, "LedColorEyes");
        Bind(led_color_mouth, "LedColorMouth");
        Bind(time_window, "TimeWindow");
        Bind(torque, "Torque");
        Bind(control_mode, "ControlMode"); // Add this binding
        
        force_output.set(100);
        torque.resize(19);
        torque.set(1); // Enable torque by default
        

        start_time = std::time(nullptr);
        goal_time_out = 5;
        firstTick = true;
        obstacle = false;
        obstacle_time = 0;
        goal_reached = false;
        previous_position.set_name("PreviousPosition");
        position_margin = 3;
        led_intensity.set(0.5);
        led_color_eyes.set(1);
        led_color_mouth.set(1);

        red_color_RGB = {1,0.3,0.3};
        force_output_tapped = false;

        number_deviations_per_time_window = matrix(present_position.size());
        number_deviations_per_time_window.set(0);

        // Initialize mode controller
        mode_controller = std::make_unique<ModeController>(force_output, goal_position_out, 
                                                          led_intensity, led_color_eyes, led_color_mouth);
    }

    void Tick()
    {
        if(goal_position_in.sum() >0)
            goal_position_out.copy(goal_position_in);
           
        
        // Handle mode switching
        static int last_mode = -1;
        if (control_mode.as_int() != last_mode) {
            mode_controller->SwitchMode(control_mode.as_int(), force_output, goal_position_out,
                                      led_intensity, led_color_eyes, led_color_mouth);
            last_mode = control_mode.as_int();
            Debug("ForceCheck: Switched to mode: " + std::string(mode_controller->GetCurrentModeName()));
        }
        
        if (present_position.connected() && present_position.size() > 0) {
            previous_position.copy(present_position.last());
        } else if (present_position.connected()) { // Connected but size 0
            previous_position.resize(0);
        }
        // If present_position is not connected, previous_position doesn't get updated from it.

        if (firstTick){
            // Time window to measure deviation
            time_window_start = std::chrono::steady_clock::now();

            
            if (present_position.connected() && present_position.size() > 0) { // Ensure input is connected and valid
                previous_position.copy(present_position); 
            } else if (present_position.connected()) { // Connected but size 0
                 previous_position.resize(0); // Match size if 0
            }
            // If not connected, previous_position remains uninitialized or as per its default constructor state.
            // This is handled by checks like previous_position.size() > i later.
        }
        // Check if time window has passed
        if (std::chrono::steady_clock::now() - time_window_start > std::chrono::seconds(time_window.as_int()))
        {
            time_window_start = std::chrono::steady_clock::now();
            number_deviations_per_time_window.reset(); // Reset the count of deviations per time window
            Debug("ForceCheck: Time window exceeded, resetting deviation count.");
        }


        if (!present_current.connected() || !current_prediction.connected()) {
            Warning("ForceCheck: PresentCurrent or CurrentPrediction not connected. Skipping deviation calculation.");
            // Decide if module should return or operate with default/no deviation
            if (firstTick) firstTick = false; // Still advance firstTick
            // previous_position update should still happen if present_position is connected
            if (present_position.connected() && present_position.size() > 0) {
                 previous_position.copy(present_position.last());
            } else if (present_position.connected()) {
                 previous_position.resize(0);
            }
            return; 
        }
        deviation.copy(current_prediction);      // Initialize deviation with current_prediction's values
        deviation.subtract(present_current);     // Subtract present_current from deviation in-place
                                                 // current_prediction should remain unmodified by this specific operation.
        
       


        if (present_position.connected() && goal_position_in.connected() && !firstTick){ // current_prediction and present_current checked above

            started_transition = StartedTransition(present_position, start_position, position_margin);
            goal_reached = GoalReached(present_position, goal_position_in, position_margin);

            // Debug("ForceCheck: Deviation: " + deviation.json());
            // Debug("ForceCheck: Present current: " + present_current.json());
            // Debug("ForceCheck: Current prediction: " + current_prediction.json());
            // Debug("ForceCheck: Goal position: " + goal_position_in.json());
            // Debug("ForceCheck: Start position: " + start_position.json());
            // Debug("ForceCheck: Present position: " + present_position.json());
            // Debug("ForceCheck: Previous position: " + previous_position.json());
            // Debug("ForceCheck: Force output: " + force_output.json());
            // Debug("ForceCheck: Started transition: " + started_transition.json());

            
            
            
            //find highest deviance
            double highest_deviance = 0.0;
            int highest_deviance_index = 0;
            for (int i = 0; i < deviation.size(); i++) {
                if (abs(deviation[i]) > highest_deviance && started_transition[i] == 1) {
                    highest_deviance = abs(deviation[i]);
                    highest_deviance_index = i;
                }
                if (abs(deviation[i]) > allowed_deviance[i]) {
                    number_deviations_per_time_window[i]++;
                }
            }
            double deviance_ratio = highest_deviance / allowed_deviance[highest_deviance_index];

            // Use the mode controller to handle the deviation
            mode_controller->HandleDeviation(deviation, present_position, goal_position_in,
                                             start_position, allowed_deviance, started_transition,
                                             number_deviations_per_time_window, torque, (double)time_window,
                                             (double)force_change_rate);

            // Set LED colors based on current mode and deviance
        
            double intensity = clip(deviance_ratio, 0.5, 1.0);
            led_intensity.set(intensity);
            mode_controller->SetLEDColor(deviance_ratio);
            
    
        }
        else
        {
            if (!present_position.connected())
                Warning("ForceCheck: PresentPosition not connected.");
            if (!goal_position_in.connected())
                Warning("ForceCheck: GoalPositionIn not connected.");
            if (present_position.size() == 0)
                Warning("ForceCheck: PresentPosition is empty.");
            if (goal_position_in.size() == 0)
                Warning("ForceCheck: GoalPositionIn is empty.");
        }

            started_transition = StartedTransition(present_position, start_position, position_margin);
            goal_reached = GoalReached(present_position, goal_position_in, position_margin);
            

       
        firstTick=false;
        Debug("ForceCheck: Current mode: " + std::string(mode_controller->GetCurrentModeName()) + 
              " (Index: " + std::to_string(mode_controller->GetModeIndex()) + ")");
        
        Debug("ForceCheck: Deviation: " + deviation.json());
        Debug("ForceCheck: Number deviations per time window: " + number_deviations_per_time_window.json());
    }


};


INSTALL_CLASS(ForceCheck);

