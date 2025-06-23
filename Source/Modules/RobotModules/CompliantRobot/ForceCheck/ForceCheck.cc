#include "ikaros.h"
#include <chrono>

using namespace ikaros;

// Base class for control modes
class ControlMode {
protected:
    ikaros::Component* parent_component; // Or ForceCheck*, depending on your needs
    matrix& force_output;
    matrix& goal_position_out;
    matrix& led_intensity;
    matrix& led_color_eyes;
    matrix& led_color_mouth;

public:
    ControlMode(ikaros::Component* parent, matrix& force_out, matrix& goal_out, matrix& led_int,
                matrix& led_eyes, matrix& led_mouth)
        : parent_component(parent), force_output(force_out), goal_position_out(goal_out),
          led_intensity(led_int), led_color_eyes(led_eyes), led_color_mouth(led_mouth) {}

    virtual ~ControlMode() = default;
    virtual void HandleDeviation(matrix& deviation, matrix& present_position,
                                matrix& goal_position_in, matrix& start_position,
                                matrix& allowed_deviance, matrix& started_transition,
                                matrix& number_deviations_per_time_window, matrix& torque,
                                double time_window, double pullback_amount) = 0;
    virtual void SetLEDColor(double deviance_ratio) = 0;
    virtual const char* GetModeName() const = 0;
};

// Normal pose tracking mode
class NormalMode : public ControlMode {
private:
    bool obstacle_detected = false; // Flag to indicate if an obstacle is detected
    std::chrono::steady_clock::time_point last_obstacle_time;
    matrix halt_goal_position; // Stores the halt goal position
public:
    NormalMode(ikaros::Component* parent, matrix& force_out, matrix& goal_out, matrix& led_int,
               matrix& led_eyes, matrix& led_mouth)
        : ControlMode(parent, force_out, goal_out, led_int, led_eyes, led_mouth) {}

    void HandleDeviation(matrix& deviation, matrix& present_position,
                        matrix& goal_position_in, matrix& start_position,
                        matrix& allowed_deviance, matrix& started_transition,
                        matrix& number_deviations_per_time_window, matrix& torque,
                        double time_window, double pullback_amount) override {

        if (present_position.size() == 0 || goal_position_in.size() == 0 || goal_position_out.size() == 0)
        {
            parent_component->Warning("Normal Mode: Empty input matrices (present_position or goal_position_in), skipping HandleDeviation.");
            if (goal_position_out.size() > 0) goal_position_out.reset(); // Still ensure output is reset if possible
            return;
        }
        goal_position_out.reset(); // Initialize goal_position_out to zero
        
        bool an_obstacle_requires_active_avoidance_this_tick = false;
        halt_goal_position.copy(goal_position_in); // Start with the input goal position
        
        for (int i = 0; i < deviation.size(); i++) {
            if (number_deviations_per_time_window(i) > 2) { // This threshold could be a parameter
                an_obstacle_requires_active_avoidance_this_tick = true;
                halt_goal_position(i) = present_position(i); 
            }
        }

        if (an_obstacle_requires_active_avoidance_this_tick){
            obstacle_detected = true;
            last_obstacle_time = std::chrono::steady_clock::now();
            goal_position_out.copy(halt_goal_position); 
            parent_component->Debug("NormalMode: Obstacle detected, halting affected motors.");
        } else {
            if (obstacle_detected) { 
                if (std::chrono::steady_clock::now() - last_obstacle_time > std::chrono::milliseconds(1000)) { 
                    obstacle_detected = false; 
                    // goal_position_out is already goal_position_in (pass-through by default)
                    parent_component->Debug("NormalMode: Obstacle detection timed out, resuming normal goal tracking.");
                } else {
                    // Still in timeout period, maintain halt if it was set
                    goal_position_out.copy(halt_goal_position);
                }
            }
            // If obstacle_detected was false, goal_position_out remains goal_position_in (pass-through).
        }
    }
    
    void SetLEDColor(double deviance_ratio) override {
        matrix color(3);
        
        if (obstacle_detected) { // Prioritize red if obstacle is actively being handled
            // Red for high deviation/obstacle
            color(0) = 1.0;
            color(1) = 0.0; 
            color(2) = 0.0;
        }
        else if (deviance_ratio > 0.8) {
            // Orange for medium deviances
            color(0) = 1.0; // Red
            color(1) = 0.5; // Green
            color(2) = 0.0; // Blue
        }
        else 
        {
            // White for normal operation
            color.set(1.0);
        }

            // Apply to all LEDs
            for (int i = 0; i < 12; i++)
            {
                led_color_eyes(0, i) = color(0);
                led_color_eyes(1, i) = color(1);
                led_color_eyes(2, i) = color(2);
            }
            for (int i = 0; i < 8; i++)
            {
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
    bool obstacle_detected = false; // Renamed for consistency, represents active avoidance state
    matrix avoidance_goal_position; 
    std::chrono::steady_clock::time_point last_avoidance_trigger_time;
public:
    AvoidantMode(ikaros::Component* parent, matrix& force_out, matrix& goal_out, matrix& led_int, 
                 matrix& led_eyes, matrix& led_mouth)
        : ControlMode(parent, force_out, goal_out, led_int, led_eyes, led_mouth) {}
    
    void HandleDeviation( matrix& deviation,  matrix& present_position,
                        matrix& goal_position_in,  matrix& start_position,
                        matrix& allowed_deviance,  matrix& started_transition,
                        matrix& number_deviations_per_time_window, matrix& torque,  
                        double time_window, double pullback_amount) override {
        if (present_position.size() == 0 || goal_position_in.size() == 0 || goal_position_out.size() == 0) {
            parent_component->Warning("AvoidantMode: Empty input matrices, skipping HandleDeviation.");
            return;
        }

        goal_position_out.reset(); // Initialize goal_position_out to zero
        
        
        bool an_obstacle_requires_active_avoidance_this_tick = false;
        
        
        if (!obstacle_detected) {
            avoidance_goal_position.copy(goal_position_in); // Start with the input goal position
            for (int i = 0; i < deviation.size(); i++)
            {
                // Condition for active avoidance: high deviation count OR if obstacle_detected is already true (meaning a fast push likely triggered this mode)
                if (number_deviations_per_time_window[i] > 1)
                { // This threshold could be a parameter
                    double movement_direction = (deviation(i) - deviation.last()[i] > 0) ? pullback_amount : -pullback_amount;

                    
                    
                    avoidance_goal_position(i) = present_position(i) + movement_direction;
                    an_obstacle_requires_active_avoidance_this_tick = true;
                    parent_component->Debug("AvoidantMode: Motor " + std::to_string(i) + " actively avoiding, goal changed by " + std::to_string(movement_direction) + " degrees.");
                }
            }
        }
        

        if (an_obstacle_requires_active_avoidance_this_tick) {
            obstacle_detected = true;
            last_avoidance_trigger_time = std::chrono::steady_clock::now();
            goal_position_out.copy(avoidance_goal_position); 

        } else {
            if (obstacle_detected) { 
                if (std::chrono::steady_clock::now() - last_avoidance_trigger_time > std::chrono::milliseconds(1500)) { // Timeout for avoidance state
                    obstacle_detected = false; 
                    parent_component->Debug("AvoidantMode: Avoidance state timed out, resuming normal goal tracking.");
                    // goal_position_out is already goal_position_in (pass-through by default)
                } else {
                    // Still in avoidance timeout, maintain avoidance goal
                    goal_position_out.copy(avoidance_goal_position);
                }
            }
            // If obstacle_detected was false, goal_position_out remains goal_position_in (pass-through).
        }
    }
    
    void SetLEDColor(double deviance_ratio) override {
        matrix color(3);
        if (obstacle_detected) { // If actively avoiding
            // Red when an obstacle is detected / actively avoiding
            color(0) = 1.0; color(1) = 0.0; color(2) = 0.0;
        } else if (deviance_ratio > 0.7) {
            // Orange for high deviation (but not actively avoiding)
            color(0) = 1.0; color(1) = 0.5; color(2) = 0.0;
        }
        else {
            // Yellow for base avoidance mode color (when not actively avoiding but still in this mode)
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
    bool torque_disabled =false; // True if torque is currently set to 0
    bool high_deviance_detected = false; // True if high deviance was detected in this mode
    std::chrono::steady_clock::time_point time_of_stabilisation; // Managed internally
    bool get_stabilisation_time = true; // Managed internally
    bool compliance_mode_active; // True if conditions for compliance were met and torque was disabled
    bool high_deviance_detected_internally = false; // Internal flag for this mode's logic
    matrix last_stable_position; // Used for stabilization check
    matrix previous_position; // Used to track last position for stability checks

public:
    CompliantMode(ikaros::Component* parent, matrix& force_out, matrix& goal_out, matrix& led_int, 
                  matrix& led_eyes, matrix& led_mouth)
        : ControlMode(parent, force_out, goal_out, led_int, led_eyes, led_mouth), 
          torque_disabled(false), compliance_mode_active(false) {}
    
    void HandleDeviation(matrix& deviation, matrix& present_position,
                        matrix& goal_position_in, matrix& start_position,
                        matrix& allowed_deviance, matrix& started_transition,
                        matrix& number_deviations_per_time_window, matrix& torque,
                        double time_window, double pullback_amount) override {
        if (present_position.size() == 0 || goal_position_in.size() == 0) {
             parent_component->Warning("CompliantMode: Empty input matrices, skipping HandleDeviation.");
            return;
        }



        // This mode is entered if "being held" is detected by ForceCheck's main Tick.
        // Its primary job here is to manage torque disable/re-enable.
        // It can also use number_deviations_per_time_window as an additional trigger for torque disable.
        previous_position.copy(present_position.last());
        for (int i = 0; i < deviation.size(); i++)
        {
            if (number_deviations_per_time_window[i] > 3)
            {
                high_deviance_detected = true;
                break;
            }
        }

        if (high_deviance_detected && !torque_disabled)
        {
            torque_disabled = true;
            compliance_mode_active = true;
            torque.set(0); // Disable torque
        }
        else if (compliance_mode_active && torque_disabled)
        {
            // Check if position has stabilized to re-enable torque
            int position_margin = 3; // Margin for position stability
            static int stable_count = 0;

            bool all_motors_stable = true;
            for (int i = 0; i < present_position.size(); i++)
            {
                if (abs(present_position(i) - previous_position(i)) > position_margin)
                {
                    all_motors_stable = false;
                    break;
                }
            }

            if (all_motors_stable)
            {
                stable_count++;
            }
            else
            {
                stable_count = 0; // Reset if any motor deviates
            }

            parent_component->Debug("ForceCheck: Compliance motor, stability count: " + std::to_string(stable_count));

            if (stable_count >= 30)
            { // 100 counts for any motor (arbitary number)
                if (get_stabilisation_time)
                {
                    time_of_stabilisation = std::chrono::steady_clock::now();
                    get_stabilisation_time = false; // Set to false to avoid resetting time
                }

                parent_component->Debug("ForceCheck: Compliance motor stable duration: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time_of_stabilisation).count()));
                if (std::chrono::steady_clock::now() - time_of_stabilisation > std::chrono::seconds(1))
                {
                    parent_component->Debug("ForceCheck: Compliance motor re-enabling torque after stabilization.");
                    torque.set(1);
                    torque_disabled = false;
                    high_deviance_detected = false; // Reset high deviance flag
                    stable_count = 0;
                    get_stabilisation_time = true; // Reset to allow future stabilization checks
                }
            }
            else
            {
                get_stabilisation_time = true; // Reset the flag if not stable
            }
        }

        if (!torque_disabled && compliance_mode_active)
        {
            // Follow the external force by updating goal to current position
            compliance_mode_active = false; // Reset compliance mode after following
            get_stabilisation_time = true;  // Reset to allow future stabilization checks
        }
    }

    void SetLEDColor(double deviance_ratio) override {
        // Green for compliant mode
        matrix color(3);
        if (torque_disabled) { // If torque is actively disabled by this mode
            // Intense green
            color(0) = 0.0; // Red
            color(1) = 1.0; // Green
            color(2) = 0.0; // Blue
        } 
         else {
            // Default color when in Compliant mode but not actively disabling torque (e.g. just entered mode)
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
    ikaros::Component* parent_component;
    // Store references passed from ForceCheck
    matrix& p_force_output;
    matrix& p_goal_position_out;
    matrix& p_led_intensity;
    matrix& p_led_color_eyes;
    matrix& p_led_color_mouth;


public:
    ModeController(ikaros::Component* parent, matrix& force_out, matrix& goal_out, matrix& led_int,
                   matrix& led_eyes, matrix& led_mouth) 
        : mode_index(0), parent_component(parent),
          p_force_output(force_out), p_goal_position_out(goal_out), p_led_intensity(led_int),
          p_led_color_eyes(led_eyes), p_led_color_mouth(led_mouth) {
        // Start with normal mode
        current_mode = std::make_unique<NormalMode>(parent_component, p_force_output, p_goal_position_out, p_led_intensity, p_led_color_eyes, p_led_color_mouth);
    }

    // SwitchMode no longer needs matrix refs, as they are stored from constructor
    void SwitchMode(int new_mode) {
        if (new_mode == mode_index && current_mode) { // No change or already initialized
            // parent_component->Debug("ModeController: Mode " + std::to_string(new_mode) + " already active.");
            return;
        }
        mode_index = new_mode % 3; // 0=Normal, 1=Avoidant, 2=Compliant

        parent_component->Debug("ModeController: Switching to mode index " + std::to_string(mode_index));
        switch (mode_index) {
        case 0:
            current_mode = std::make_unique<NormalMode>(parent_component, p_force_output, p_goal_position_out, p_led_intensity, p_led_color_eyes, p_led_color_mouth);
            break;
        case 1:
            current_mode = std::make_unique<AvoidantMode>(parent_component, p_force_output, p_goal_position_out, p_led_intensity, p_led_color_eyes, p_led_color_mouth);
            break;
        case 2:
            current_mode = std::make_unique<CompliantMode>(parent_component, p_force_output, p_goal_position_out, p_led_intensity, p_led_color_eyes, p_led_color_mouth);
            break;
        default:
            parent_component->Warning("ModeController: Unknown mode index " + std::to_string(mode_index) + ", defaulting to NormalMode.");
            current_mode = std::make_unique<NormalMode>(parent_component, p_force_output, p_goal_position_out, p_led_intensity, p_led_color_eyes, p_led_color_mouth);
            mode_index = 0;
            break;
        }
        parent_component->Debug("ModeController: Switched to mode: " + std::string(current_mode->GetModeName()));
    }
    void HandleDeviation(matrix& deviation, matrix& present_position,
                        matrix& goal_position_in, matrix& start_position,
                        matrix& allowed_deviance, matrix& started_transition,
                        matrix& number_deviations_per_time_window, matrix& torque,
                        double time_window, double pullback_amount) {
        if (!current_mode) {
            parent_component->Error("ModeController: current_mode is null in HandleDeviation!");
            SwitchMode(0); // Attempt to recover by switching to NormalMode
        }
        current_mode->HandleDeviation(deviation, present_position, goal_position_in,
                                    start_position, allowed_deviance, started_transition,
                                    number_deviations_per_time_window, torque, time_window, pullback_amount);
    }
    
    void SetLEDColor(double deviance_ratio) {
        if (!current_mode) {
            parent_component->Error("ModeController: current_mode is null in SetLEDColor!");
            return;
        }
        current_mode->SetLEDColor(deviance_ratio);
    }
    
    const char* GetCurrentModeName() const {
        if (!current_mode) return "None (Error)";
        return current_mode->GetModeName();
    }
    
    int GetModeIndex() const { return mode_index; }
};

class ForceCheck: public Module
{
public: // Ensure INSTALL_CLASS can access constructor if it's implicitly used.
    //parameters
    parameter pullback_amount;
    parameter time_window;
    parameter control_mode; 
    
    // New parameters for automatic mode switching
    parameter automatic_mode_switching_enabled;
    parameter sustained_hold_duration_ms;
    parameter fast_push_deviation_increase;
    parameter max_movement_degrees_when_held;
    parameter obstacle_detection_deviation_count;
    parameter compliant_trigger_min_deviation_ratio;

    //inputs
    matrix present_current;
    matrix current_limit;
    matrix present_position;// assumes degrees
    matrix goal_position_in;
    matrix start_position;
    matrix allowed_deviance_input;

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
    matrix motor_in_motion;
    matrix allowed_deviance;
   
    bool firstTick;
    // bool obstacle; // Managed within modes or by new logic
    // double obstacle_time; // Managed within modes or by new logic
    bool goal_reached;
    double start_time;
    int goal_time_out;
    bool force_output_tapped;
    std::chrono::steady_clock::time_point force_output_tapped_time_point;
    std::chrono::steady_clock::time_point time_window_start;

    // Add mode controller
    std::unique_ptr<ModeController> mode_controller;

    // New internal variables for auto mode switching
    std::chrono::steady_clock::time_point sustained_high_dev_start_time;
    bool evaluating_sustained_hold;


        double
        Tapering(double error, double threshold)
    {
        if (abs(error) < threshold)
            return sin(M_PI_2)*(abs(error)/threshold);
        else
            return 1;
    }
    
  



    bool GoalReached(matrix present_position, matrix goal_position, int margin)
    {
        if (present_position.size() != goal_position.size() || present_position.size() == 0) return false;
        for (int i = 0; i < present_position.size(); i++) {
            // Assuming positions are comparable types (e.g. both int or double after casting if needed)
            if (abs( (double)present_position[i] - (double)goal_position[i]) > margin)
                return false;
        }
        return true;
    }

    matrix StartedTransition(matrix& present_position, matrix& start_position, int margin)
    {
        if (present_position.size() == 0 || present_position.size() != start_position.size())
        {
            Warning("ForceCheck: StartedTransition - Input matrix size mismatch or empty. Returning empty matrix.");
            return matrix(0); 
        }

        matrix transition_status(deviation.size()); // Assuming deviation is correctly sized
        transition_status.set(0.0); // Initialize to not started

        for (int i = 0; i < deviation.size() && i < present_position.size() && i < start_position.size() && i < present_current.size(); i++){
            if (abs( (double)start_position(i) - (double)present_position(i)) < margin || abs( (double)present_current(i))<5)
                transition_status(i) = 0.0; // Not started
            else
                transition_status(i) = 1.0; // Started
        }
        
        return transition_status;
    }

    matrix MotorInMotion(matrix& present_position, matrix& goal_position, int margin)
    {
        if (present_position.size() == 0 || present_position.size() != goal_position.size())
        {
            Warning("ForceCheck: MotorInMotion - Input matrix size mismatch or empty. Returning empty matrix.");
            return matrix(0); 
        }

        matrix in_motion(deviation.size()); // Assuming deviation is correctly sized
      
        matrix previous_position = present_position.last(); // Get the last position for comparison

        for (int i = 0; i < deviation.size(); i++) {
            if (abs( (double)present_position(i) - (double)previous_position(i)) > 1)
                in_motion(i) = 1.0; // In motion
            else
                in_motion(i) = 0.0; // Not in motion
        }
        
        return in_motion;
    
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
        Bind(allowed_deviance_input, "AllowedDeviance");
        Bind(pullback_amount, "PullBackAmount");
        Bind(led_intensity, "LedIntensity");
        Bind(led_color_eyes, "LedColorEyes");
        Bind(led_color_mouth, "LedColorMouth");
        Bind(time_window, "TimeWindow");
        Bind(torque, "Torque");
        Bind(control_mode, "ControlMode"); 
        
        // Bind new parameters
        Bind(automatic_mode_switching_enabled, "AutomaticModeSwitchingEnabled");
        Bind(sustained_hold_duration_ms, "SustainedHoldDurationMs");
        Bind(fast_push_deviation_increase, "FastPushDeviationIncrease");
        Bind(max_movement_degrees_when_held, "MaxMovementDegreesWhenHeld"); // Degrees per tick
        Bind(obstacle_detection_deviation_count, "ObstacleDetectionDeviationCount"); // (e.g. >2)
        Bind(compliant_trigger_min_deviation_ratio, "CompliantTriggerMinDeviationRatio");


        torque.set(1); // Enable torque by default

        start_time = std::time(nullptr); // Using C time, consider chrono if more precision needed elsewhere
        goal_time_out = 5; // Seconds
        firstTick = true;
        // obstacle = false; // Handled by mode logic or new auto-switch logic
        // obstacle_time = 0; // Handled by mode logic
        goal_reached = false;
        previous_position.set_name("PreviousPosition");
  
        position_margin = 3; // Degrees
        led_intensity.set(0.5); // Default intensity
        led_color_eyes.set(1); // Default white
        led_color_mouth.set(1); // Default white

        red_color_RGB = {1,0.3,0.3}; // Example, not directly used if SetLEDColor in modes handles all
        force_output_tapped = false; // Unused?
        force_output_tapped_time_point = std::chrono::steady_clock::now(); // Unused?

        number_deviations_per_time_window.set_name("NumberDeviationsPerTimeWindow");
        number_deviations_per_time_window.copy(deviation); // Ensure size matches present_position
        number_deviations_per_time_window.reset(); // Initialize to zero

        evaluating_sustained_hold = false;
        time_window_start = std::chrono::steady_clock::now();
        allowed_deviance.copy(allowed_deviance_input); // Initialize allowed_deviance with input

        // Initialize mode controller
        // Ensure output matrices are valid before passing if their size depends on inputs
        // For now, assume they are or will be handled by ModeController/Modes
        mode_controller = std::make_unique<ModeController>(this, force_output, goal_position_out, 
                                                          led_intensity, led_color_eyes, led_color_mouth);
        
        // Ensure torque is initialized with correct size and default value
        // This might need to happen after first Tick or when present_position is sized
        // For now, let's assume a fixed size or handle resize in Tick.
        // If using a typical robot setup, you might know the number of servos.
        // Example: if 19 servos:
        // torque.resize(19); torque.set(1);
    }
    
    const char* GetModeNameByIndex(int index) {
        if (index == 0) return "Normal";
        if (index == 1) return "Avoidant";
        if (index == 2) return "Compliant";
        return "Unknown";
    }




    void Tick()
    {
      
        if (present_current.size() == 0 || present_position.size() == 0 || goal_position_in.size() == 0 ||
            start_position.size() == 0 || allowed_deviance.size() == 0) {
            Warning("ForceCheck: One or more input matrices are empty, skipping Tick logic.");
            return; // Skip Tick logic if inputs are not ready
        }
        if (firstTick) {
            previous_position.copy(present_position); // Initialize previous_position on first tick
            firstTick = false; // Set to false after first initialization
        }


        // Handle manual mode switching via parameter
        static int last_manual_mode_setting = -1;
        if (control_mode.as_int() != last_manual_mode_setting) {
            if (!automatic_mode_switching_enabled || last_manual_mode_setting == -1) { // Allow manual override or initial set
                mode_controller->SwitchMode(control_mode.as_int());
                Debug("ForceCheck: Manually switched to mode: " + std::string(mode_controller->GetCurrentModeName()));
            }
            last_manual_mode_setting = control_mode.as_int();
        }
        
        // Check if time window has passed for deviation counting
        if (std::chrono::steady_clock::now() - time_window_start > std::chrono::seconds(time_window.as_int()))
        {
            time_window_start = std::chrono::steady_clock::now();
            if (number_deviations_per_time_window.size() > 0) number_deviations_per_time_window.reset();
            Debug("ForceCheck: Time window exceeded, resetting deviation count.");
        }

        if (!present_current.connected() || !current_prediction.connected() || 
            present_current.size() == 0 || current_prediction.size() == 0 ||
            present_current.size() != current_prediction.size()) {
            Warning("ForceCheck: PresentCurrent or CurrentPrediction not connected, empty, or mismatched sizes. Skipping deviation calculation and mode logic.");
            if (deviation.size() > 0) deviation.reset(); // Clear deviation if it can't be calculated
            if (firstTick) firstTick = false; 
            if (present_position.connected() && present_position.size() > 0) {
                previous_position.copy(present_position); // Keep previous_position updated
            }
    



            if (mode_controller && present_position.connected() && goal_position_in.connected() && start_position.connected() && allowed_deviance.connected()) {
                 mode_controller->HandleDeviation(deviation, present_position, goal_position_in,
                                             start_position, allowed_deviance, started_transition, // started_transition might be empty
                                             number_deviations_per_time_window, torque, (double)time_window,
                                             (double)pullback_amount);
                 if (allowed_deviance.size() > 0 && allowed_deviance[0] > 0) { // Avoid div by zero
                    mode_controller->SetLEDColor(0.0); // No real deviation, so 0 ratio
                 } else {
                    mode_controller->SetLEDColor(0.0);
                 }
            }
            return; 
        }
        

        deviation.copy(current_prediction);
        deviation.subtract(present_current);
        Debug("ForceCheck: Deviation calculated: " + deviation.json());
        
        if (goal_position_in.connected() && !firstTick && allowed_deviance.connected() && allowed_deviance.size() > 0 && start_position.connected()) {
            started_transition = StartedTransition(present_position, start_position, position_margin);
            goal_reached = GoalReached(present_position, goal_position_in, position_margin);
            motor_in_motion = MotorInMotion(present_position, goal_position_in, position_margin);



            // Update number_deviations_per_time_window
            for (int i = 0; i < allowed_deviance_input.size(); i++) {
                
                if (motor_in_motion[i] == 1) {
                    allowed_deviance(i) = allowed_deviance_input(i); // Use input allowed_deviance if motor is in motion
                }
                else {
                    allowed_deviance(i) = allowed_deviance_input(i)/3; // If not in motion, set to 0
                }
                
                Debug("ForceCheck: Motor " + std::to_string(i) + " deviation: " + std::to_string(deviation[i]) + 
                      ", threshold: " + std::to_string(allowed_deviance(i)) + 
                      ", Motor moving: " + std::to_string(motor_in_motion[i]));
                if (abs(deviation[i]) > allowed_deviance(i)) {
                    number_deviations_per_time_window[i]++;
                }
            }

            // Automatic Mode Switching Logic
            if (automatic_mode_switching_enabled) {
                bool fast_push_detected_this_tick = false;
                bool being_held_detected_this_tick = false;
                bool obstacle_inferred_this_tick = false;

                // 1. Detect Fast Push
               
                for (int i = 0; i < deviation.size(); i++) {
                    double dev_change = abs(deviation[i]) - abs(deviation.last()[i]); // Assuming deviation.last(i) gives the previous tick's deviation
                    if (dev_change > fast_push_deviation_increase.as_int()) {
                        fast_push_detected_this_tick = true;
                        Debug("ForceCheck: Fast push detected for motor " + std::to_string(i) + ", dev_change: " + std::to_string(dev_change));
                        break;
                    }
                }
                

                // 2. Detect Sustained Hold
                bool high_dev_and_low_movement_this_tick = false;
                for (int i = 0; i < deviation.size(); i++) {
                    if (abs(deviation(i)) > allowed_deviance(i) &&
                        abs( (double)present_position(i) - present_position.last()[i]) < max_movement_degrees_when_held.as_int()) {
                        high_dev_and_low_movement_this_tick = true;
                        Debug("ForceCheck: Motor " + std::to_string(i) + " has high dev and low movement.");
                        break;
                    }
                }
                

                if (high_dev_and_low_movement_this_tick) {
                    if (!evaluating_sustained_hold) {
                        evaluating_sustained_hold = true;
                        sustained_high_dev_start_time = std::chrono::steady_clock::now();
                        Debug("ForceCheck: Started evaluating sustained hold.");
                    } else {
                        auto hold_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - sustained_high_dev_start_time);
                        if (hold_duration.count() > sustained_hold_duration_ms.as_int()) {
                            being_held_detected_this_tick = true;
                            Debug("ForceCheck: Sustained hold detected. Duration: " + std::to_string(hold_duration.count()) + "ms");
                        }
                    }
                } else {
                    if (evaluating_sustained_hold) Debug("ForceCheck: High dev and low movement condition ended. Resetting sustained hold eval.");
                    evaluating_sustained_hold = false;
                }

                // 3. Detect Obstacle Inference
                for (int i = 0; i < number_deviations_per_time_window.size(); i++) {
                    if (number_deviations_per_time_window[i] > obstacle_detection_deviation_count.as_int()) {
                        obstacle_inferred_this_tick = true;
                        Debug("ForceCheck: Obstacle inferred for motor " + std::to_string(i) + ", num_devs: " + std::to_string(number_deviations_per_time_window[i]));
                        break;
                    }
                }

                // Determine Desired Mode based on priority
                int current_mode_idx = mode_controller->GetModeIndex();
                int desired_mode_idx = current_mode_idx;

                if (fast_push_detected_this_tick) {
                    desired_mode_idx = 1; // AvoidantMode
                } else if (being_held_detected_this_tick) {
                    desired_mode_idx = 2; // CompliantMode
                } else if (obstacle_inferred_this_tick) {
                    desired_mode_idx = 0; // NormalMode
                }
                Debug("ForceCheck: Desired mode index: " + std::to_string(desired_mode_idx) + 
                      " (Current mode index: " + std::to_string(current_mode_idx) + ")");
                // If no condition met, desired_mode_idx remains current_mode_idx

                if (desired_mode_idx != current_mode_idx) {
                    Debug("ForceCheck: Auto switching: Current '" + std::string(mode_controller->GetCurrentModeName()) + 
                          "' to Desired '" + GetModeNameByIndex(desired_mode_idx) + "'.");
                    control_mode = desired_mode_idx; // This will be picked up by the manual switch logic at the start of next tick
                    mode_controller->SwitchMode(desired_mode_idx); // Switch immediately
                    last_manual_mode_setting = desired_mode_idx; // Update to prevent immediate manual revert if param was different
                }
            }
       


            // Find highest deviance for LED intensity and color calculation
            double highest_deviance_abs = 0.0;
            int highest_deviance_index = 0;
            if (deviation.size() > 0) { // Ensure deviation is not empty
                for (int i = 0; i < deviation.size(); i++) {
                    if (abs(deviation[i]) > highest_deviance_abs) {
                        highest_deviance_abs = abs(deviation[i]);
                        highest_deviance_index = i;
                    }
                }
            }
            
            double deviance_ratio = 0.0;
            if (allowed_deviance.size() > highest_deviance_index && allowed_deviance[highest_deviance_index] > 0) {
                 deviance_ratio = highest_deviance_abs / allowed_deviance[highest_deviance_index];
            } else if (allowed_deviance.size() > 0 && allowed_deviance[0] > 0) { // Fallback if index is bad but allowed_deviance exists
                 deviance_ratio = highest_deviance_abs / allowed_deviance[0];
            }


            // Use the mode controller to handle the deviation
            mode_controller->HandleDeviation(deviation, present_position, goal_position_in,
                                            start_position, allowed_deviance, started_transition,
                                            number_deviations_per_time_window, torque, (double)time_window,
                                            (double)pullback_amount);

            // Set LED colors based on current mode and deviance
            double intensity = clip(deviance_ratio * 0.5 + 0.5, 0.5, 1.0); // Scale ratio so 0 -> 0.5 intensity, 1 -> 1.0 intensity
            led_intensity.set(intensity);
            mode_controller->SetLEDColor(deviance_ratio);
        }
        else // Conditions for full operation not met (e.g. firstTick, missing connections)
        {
            if (firstTick) {
                 // Initialize previous_position if possible
                if (present_position.connected() && present_position.size() > 0) {
                    previous_position.copy(present_position);
                }
            } else { // Not first tick, but some other condition failed
                if (!goal_position_in.connected()) Warning("ForceCheck: GoalPositionIn not connected.");
                if (!allowed_deviance.connected() || allowed_deviance.size() == 0) Warning("ForceCheck: AllowedDeviance not connected or empty.");
                // Potentially set outputs to a safe default state
                // goal_position_out.copy(goal_position_in); // Pass through if possible
                // torque.set(1); // Ensure torque is on
                // led_intensity.set(0.5); led_color_eyes.set(1); led_color_mouth.set(1); // Default LEDs
            }
        }
        
        if (present_position.connected() && present_position.size() > 0) {
            previous_position.copy(present_position); // Update for next tick's "held" detection
        }
        firstTick=false;

        // Debug("ForceCheck: Current mode: " + std::string(mode_controller->GetCurrentModeName()) + 
        //       " (Index: " + std::to_string(mode_controller->GetModeIndex()) + ")");
        // Debug("ForceCheck: Deviation: " + deviation.json()); // Can be very noisy
        // Debug("ForceCheck: NumDevs: " + number_deviations_per_time_window.json()); // Can be very noisy
        // Debug("ForceCheck: Torque: " + torque.json());
        // Debug("ForceCheck: GoalOut: " + goal_position_out.json());
    }
};

INSTALL_CLASS(ForceCheck);

