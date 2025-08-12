#include "ikaros.h"
#include <chrono>
#include <algorithm>
#include <cmath>

using namespace ikaros;

// Helper struct for deviation statistics
struct DeviationStats {
    double mean = 0.0;
    double stddev = 0.0;
    double exceed_ratio = 0.0;
    int exceed_count = 0;
    int total_samples = 0;
    
    DeviationStats(const matrix& deviation_history, int motor_idx, double threshold, int rows = -1) {
        if (deviation_history.rows() == 0) return;
        
        int actual_rows = (rows < 0) ? deviation_history.rows() : std::min(rows, deviation_history.rows());
        double sum = 0.0, sum2 = 0.0;
        
        for (int r = 0; r < actual_rows; ++r) {
            double v = std::abs((double)deviation_history(r, motor_idx));
            sum += v;
            sum2 += v * v;
            total_samples++;
            if (v > threshold) exceed_count++;
        }
        
        if (total_samples > 0) {
            mean = sum / total_samples;
            exceed_ratio = (double)exceed_count / total_samples;
            
            if (total_samples > 1) {
                double variance = std::max(0.0, (sum2 / total_samples - mean * mean));
                stddev = std::sqrt(variance);
            }
        }
    }
};

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
                                matrix& deviation_history, matrix& torque,
                                double pullback_amount, double peak_width_tolerance,
                                double current_time) = 0;
    virtual void SetLEDColor(double deviance_ratio) = 0;
    virtual const char* GetModeName() const = 0;
    
protected:
    // Helper method to apply color to all LEDs
    void ApplyColorToAllLEDs(const matrix& color) {
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
};

// Normal pose tracking mode
class NormalMode : public ControlMode {
private:
    bool obstacle_detected = false;
    double last_obstacle_time;
    matrix halt_goal_position;
    bool should_override = false; // Clear state indicator
    
public:
    NormalMode(ikaros::Component* parent, matrix& force_out, matrix& goal_out, matrix& led_int,
               matrix& led_eyes, matrix& led_mouth)
        : ControlMode(parent, force_out, goal_out, led_int, led_eyes, led_mouth) {}

    void HandleDeviation(matrix& deviation, matrix& present_position,
                        matrix& goal_position_in, matrix& start_position,
                        matrix& allowed_deviance, matrix& started_transition,
                        matrix& deviation_history, matrix& torque,
                        double pullback_amount, double peak_width_tolerance,
                        double current_time) override {

        if (present_position.size() == 0 || goal_position_in.size() == 0 || goal_position_out.size() == 0)
        {
            parent_component->Warning("Normal Mode: Empty input matrices (present_position or goal_position_in), skipping HandleDeviation.");
            if (goal_position_out.size() > 0) goal_position_out.reset(); // Still ensure output is reset if possible
            return;
        }
       
        torque.set(1); // Set torque to 1 for normal operation
        
        bool an_obstacle_requires_active_avoidance_this_tick = false;
      
        // Compute exceedance ratio over recent history per motor
        int rows = deviation_history.rows();
        if (rows > 0) {
            for (int i = 0; i < deviation.size(); i++) {
                DeviationStats stats(deviation_history, i, allowed_deviance(i), rows);
                if (stats.exceed_ratio > 0.7 && !obstacle_detected) {
                    an_obstacle_requires_active_avoidance_this_tick = true;
                    halt_goal_position.copy(present_position);
                    break;
                }
            }
        }

        if (an_obstacle_requires_active_avoidance_this_tick){
            obstacle_detected = true;
            last_obstacle_time = current_time;
            should_override = true;
            parent_component->Debug("NormalMode: Obstacle detected, halting affected motors.");
        }  
        
        // Handle obstacle timeout
        if (obstacle_detected && (current_time - last_obstacle_time > 1.0)) { 
            obstacle_detected = false;
            should_override = false;
            parent_component->Debug("NormalMode: Obstacle detection timed out, resuming normal goal tracking.");
        }
        
        // Set output based on current state
        if (should_override) {
            goal_position_out.copy(halt_goal_position);
            parent_component->Debug("NormalMode: Override active, goal_position_out = " + goal_position_out.json());
        } else {
            goal_position_out.reset(); // Let GoalSetter handle normal operation
            parent_component->Debug("NormalMode: No override, goal_position_out after reset = " + goal_position_out.json());
        }
        parent_component->Debug("NormalMode: Present tilt position: " + std::to_string(present_position(0)));
    }
    
    void SetLEDColor(double deviance_ratio) override {
        matrix color(3);
        
        if (obstacle_detected) {
            color(0) = 1.0; color(1) = 0.0; color(2) = 0.0; // Red for obstacle
        } else {
            color.set(1.0); // White for normal operation
        }
        
        ApplyColorToAllLEDs(color);
    }

    const char* GetModeName() const override { return "Normal"; }
};

// Avoidant mode - pulls away from high deviation
class AvoidantMode : public ControlMode {
private:
    bool obstacle_detected = false; // Renamed for consistency, represents active avoidance state
    matrix avoidance_goal_position; 
    double last_avoidance_trigger_time;
public:
    AvoidantMode(ikaros::Component* parent, matrix& force_out, matrix& goal_out, matrix& led_int, 
                 matrix& led_eyes, matrix& led_mouth)
        : ControlMode(parent, force_out, goal_out, led_int, led_eyes, led_mouth) {}
    
    void HandleDeviation(matrix& deviation, matrix& present_position,
                        matrix& goal_position_in, matrix& start_position,
                        matrix& allowed_deviance, matrix& started_transition,
                        matrix& deviation_history, matrix& torque,
                        double pullback_amount, double peak_width_tolerance,
                        double current_time) override {
        if (present_position.size() == 0 || goal_position_in.size() == 0 || goal_position_out.size() == 0) {
            parent_component->Warning("AvoidantMode: Empty input matrices, skipping HandleDeviation.");
            return;
        }
        torque.set(1);

        goal_position_out.reset(); // Initialize goal_position_out to zero
        
        
        bool an_obstacle_requires_active_avoidance_this_tick = false;
        
        int rows = deviation_history.rows();
        
        if (!obstacle_detected) {
            avoidance_goal_position.copy(goal_position_in); // Start with the input goal position
            for (int i = 0; i < deviation.size(); i++) {
                double lastAbs = std::abs((double)deviation(i));
                double prevAbs = std::abs((double)deviation.last()[i]);
                
                DeviationStats stats(deviation_history, i, allowed_deviance(i), rows);
                
                // Sharp peak not sustained => trigger retract
                bool sharp_peak = (lastAbs > stats.mean + 2 * stats.stddev) && 
                                (prevAbs <= (double)allowed_deviance(i)) && 
                                (stats.exceed_ratio <= peak_width_tolerance);
                // Also avoid if frequent exceedances (obstacle-like)
                bool frequent_exceed = stats.exceed_ratio > 0.5;

                if (sharp_peak || frequent_exceed) { 
                    double movement_direction = (deviation(i) > 0.0) ? pullback_amount : -pullback_amount;
                    avoidance_goal_position(i) = present_position(i) + movement_direction;
                    an_obstacle_requires_active_avoidance_this_tick = true;
                    parent_component->Debug("AvoidantMode: Motor " + std::to_string(i) +
                        (sharp_peak ? " sharp peak" : " frequent exceed") +
                        ", goal changed by " + std::to_string(movement_direction) + " degrees.");
                }
            }
        }
        

        if (an_obstacle_requires_active_avoidance_this_tick) {
            obstacle_detected = true;
            last_avoidance_trigger_time = current_time;
            goal_position_out.copy(avoidance_goal_position); 

        } else {
            if (obstacle_detected) { 
                if (current_time - last_avoidance_trigger_time > 1.5) { // Timeout for avoidance state
                    obstacle_detected = false; 
                    parent_component->Debug("AvoidantMode: Avoidance state timed out, resuming normal goal tracking.");
                    goal_position_out.reset(); // Clear override to allow GoalSetter normal operation
                } else {
                    // Still in avoidance timeout, maintain avoidance goal
                    goal_position_out.copy(avoidance_goal_position);
                }
            } else {
                // No obstacle detected and not in avoidance state - let GoalSetter handle goal setting
                goal_position_out.reset(); // Keep output clear for normal operation
            }
        }
    }
    
    void SetLEDColor(double deviance_ratio) override {
        matrix color(3);
        if (obstacle_detected) { // If actively avoiding
            // Red when an obstacle is detected / actively avoiding
            color(0) = 1.0; color(1) = 0.0; color(2) = 0.0;
        } 
        else {
            // Yellow for base avoidance mode color (when not actively avoiding but still in this mode)
            color(0) = 1.0; color(1) = 1.0; color(2) = 0.0;
        }
        
        ApplyColorToAllLEDs(color);
    }
    
    const char* GetModeName() const override { return "Avoidant"; }
};

// Compliant mode - follows external force
class CompliantMode : public ControlMode {
private:
    bool torque_disabled =false; // True if torque is currently set to 0
    bool high_deviance_detected = false; // True if high deviance was detected in this mode
    double time_of_stabilisation; // Managed internally
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
                        matrix& deviation_history, matrix& torque,
                        double pullback_amount, double peak_width_tolerance,
                        double current_time) override {
        if (present_position.size() == 0 || goal_position_in.size() == 0) {
             parent_component->Warning("CompliantMode: Empty input matrices, skipping HandleDeviation.");
            return;
        }

        // This mode is entered if "being held" is detected by ForceCheck's main Tick.
        // Its primary job here is to manage torque disable/re-enable.
        previous_position.copy(present_position.last());

        // Determine if high deviance sustained based on deviation_history statistics
        int rows = deviation_history.rows();
        high_deviance_detected = false;
        if (rows > 0) {
            for (int i = 0; i < deviation.size(); i++) {
                DeviationStats stats(deviation_history, i, allowed_deviance(i), rows);
                if (stats.mean > (double)allowed_deviance(i) * 0.8 || stats.exceed_ratio > 0.5) {
                    high_deviance_detected = true; 
                    break;
                }
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
            int position_margin = 2; // Margin for position stability
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
            { // 30 counts for any motor (arbitary number)
                if (get_stabilisation_time)
                {
                    time_of_stabilisation = current_time;
                    get_stabilisation_time = false; // Set to false to avoid resetting time
                }

                parent_component->Debug("ForceCheck: Compliance motor stable duration: " + std::to_string((current_time - time_of_stabilisation) * 1000.0));
                if (current_time - time_of_stabilisation > 1.0)
                {
                    parent_component->Debug("ForceCheck: Compliance motor re-enabling torque after stabilization.");
                    torque.set(1);
                    torque_disabled = false;
                    high_deviance_detected = false; // Reset high deviance flag
                    stable_count = 0;
                    get_stabilisation_time = true; // Reset to allow future stabilization checks
                    goal_position_out.reset(); // Resume normal operation
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
        
        ApplyColorToAllLEDs(color);
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
                        matrix& deviation_history, matrix& torque,
                        double pullback_amount, double peak_width_tolerance = 0.1,
                        double current_time = 0.0) {
        if (!current_mode) {
            parent_component->Error("ModeController: current_mode is null in HandleDeviation!");
            SwitchMode(0); // Attempt to recover by switching to NormalMode
        }
        current_mode->HandleDeviation(deviation, present_position, goal_position_in,
                                    start_position, allowed_deviance, started_transition,
                                    deviation_history, torque, pullback_amount, peak_width_tolerance, current_time);
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
    parameter history_lenght;
    parameter control_mode; 
    
    // New parameters for automatic mode switching
    parameter automatic_mode_switching_enabled;
    parameter sustained_hold_duration_ms;
    parameter fast_push_deviation_increase;
    parameter max_movement_degrees_when_held;
    parameter obstacle_detection_deviation_count;
    parameter compliant_trigger_min_deviation_ratio;
    parameter movement_threshold_scalar;
    parameter peak_width_tolerance; // Width of the peak for sharp peak detection in percentage of devaiance history

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
    // Removed number_deviations_per_time_window (replaced by deviation_history dynamics)
    matrix motor_in_motion;
    matrix allowed_deviance;
    bool firstTick;
    matrix deviation_history; // Stores recent deviations for each motor


    bool goal_reached;
    double start_time;
    int goal_time_out;
    bool force_output_tapped;
    double force_output_tapped_time_point;
    // Removed time_window_start (no time-window logic anymore)

    // Variables for auto mode switching
    double sustained_high_dev_start_time;
    bool refractory_period = false;                      // Avoid rapid mode switching
    bool fast_push_detected = false;                     // True if a fast push was detected
    bool automatic_mode_switching_enabled_value = false; // Value of the parameter
    bool evaluating_sustained_hold = false;              // True if currently evaluating sustained hold

    

    // Add mode controller
    std::unique_ptr<ModeController> mode_controller;

   
    double refractory_start_time;


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

        for (int i = 0; i < deviation.size() ; i++){
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
            if (abs( (double)present_position(i) - (double)previous_position(i)) > 0.6)
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
        Bind(history_lenght, "HistoryLength");
        Bind(torque, "Torque");
        Bind(control_mode, "ControlMode"); 
        Bind(peak_width_tolerance, "PeakWidthTolerance"); // Width of the peak for sharp peak detection in percentage of deviance history
        // Mode switching parameters
        Bind(automatic_mode_switching_enabled, "AutomaticModeSwitchingEnabled");
        Bind(sustained_hold_duration_ms, "SustainedHoldDurationMs");
        Bind(fast_push_deviation_increase, "FastPushDeviationIncrease");
        Bind(max_movement_degrees_when_held, "MaxMovementDegreesWhenHeld"); // Degrees per tick
        Bind(obstacle_detection_deviation_count, "ObstacleDetectionDeviationCount"); // (e.g. >2)
        Bind(compliant_trigger_min_deviation_ratio, "CompliantTriggerMinDeviationRatio");
        Bind(movement_threshold_scalar, "MovementThresholdScalar"); // Scalar for movement threshold

        torque.set(1); // Enable torque by default

        start_time = std::time(nullptr); // Using C time, consider chrono if more precision needed elsewhere
        goal_time_out = 5; // Seconds
        firstTick = true;
        goal_reached = false;
        previous_position.set_name("PreviousPosition");
  
        position_margin = 1; // Degrees
        led_intensity.set(0.5); // Default intensity
        led_color_eyes.set(1); // Default white
        led_color_mouth.set(1); // Default white

        red_color_RGB = {1,0.3,0.3}; // Example, not directly used if SetLEDColor in modes handles all
        force_output_tapped = false; // Unused?
        force_output_tapped_time_point = GetTime(); // Unused?

        // Removed number_deviations_per_time_window init and time_window_start since we're using deviation_history
        
        deviation_history = matrix(0, deviation.size()); // Store recent deviations for each motor
        deviation_history.set_name("DeviationHistory");


        evaluating_sustained_hold = false;
        allowed_deviance.copy(allowed_deviance_input); // Initialize allowed_deviance with input

        // Initialize mode controller
        mode_controller = std::make_unique<ModeController>(this, force_output, goal_position_out, 
                                                          led_intensity, led_color_eyes, led_color_mouth);
        
        // Ensure torque is initialized with correct size and default value
        // This might need to happen after first Tick or when present_position is sized
        // For now, let's assume a fixed size or handle resize in Tick.
    }
    
    const char* GetModeNameByIndex(int index) {
        if (index == 0) return "Normal";
        if (index == 1) return "Avoidant";
        if (index == 2) return "Compliant";
        return "Unknown";
    }




    void Tick()
    {
        // Print("Time: " + std::to_string(GetTime()));
        // Print("Nominal Time: " + std::to_string(GetNominalTime()));
        // Print("Real Time: " + std::to_string(GetRealTime()));
        deviation.copy(current_prediction);
        deviation.subtract(present_current);

        // Maintain deviation history as an ever-growing buffer; we will compute stats over the last HistoryLength samples
        if (deviation_history.cols() != deviation.size() || deviation_history.rows() >= history_lenght.as_int())
        {
            deviation_history.resize(0, deviation.size());
        }

        if (deviation.size() > 0)
        {
            deviation_history.push(deviation, true); // Append current deviation snapshot
            //deviation_history.print();               // Debug print of deviation history
        }

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
        
        // Remove time-window based reset, we rely on deviation_history dynamics now

        //Check refractory period for automatic mode switching
        if (GetTime() - refractory_start_time > 2.0) {
            refractory_period = false; // Reset refractory period after 2 seconds

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
    
        }
        
        if (goal_position_in.connected() && !firstTick && allowed_deviance.connected() && allowed_deviance.size() > 0 && start_position.connected()) {
            started_transition = StartedTransition(present_position, start_position, position_margin);
            goal_reached = GoalReached(present_position, goal_position_in, position_margin);
            motor_in_motion = MotorInMotion(present_position, goal_position_in, position_margin);

            // Dynamically adapt allowed deviance for motors in motion
            for (int i = 0; i < allowed_deviance_input.size(); i++) {
                if (motor_in_motion[i] == 1) {
                    allowed_deviance(i) = allowed_deviance_input(i) * movement_threshold_scalar;
                    // Check if motor is within 5 degrees of start position
                    if (abs((double)present_position(i) - (double)start_position(i)) <= 10.0) {
                        allowed_deviance(i) = allowed_deviance_input(i) * 1.5; // Increase by 50%
                    }
                }
                else {
                    allowed_deviance(i) = allowed_deviance_input(i); // More sensitive when not in motion
                }
               
                Debug("ForceCheck: Motor " + std::to_string(i) + " deviation: " + std::to_string(deviation[i]) + 
                      ", threshold: " + std::to_string(allowed_deviance(i)) + 
                      ", Motor moving: " + std::to_string(motor_in_motion[i]));
            }
            
            // Use the mode controller to handle the deviation with history-driven dynamics
            double current_time = GetTime();
            mode_controller->HandleDeviation(deviation, present_position, goal_position_in,
                                             start_position, allowed_deviance, started_transition,
                                             deviation_history, torque, (double)pullback_amount, (double) peak_width_tolerance, current_time);
            if (allowed_deviance.size() > 0 && allowed_deviance[0] > 0)
            {                                      // Avoid div by zero
                mode_controller->SetLEDColor(0.0); // No real deviation, so 0 ratio
            }
            else
            {
                mode_controller->SetLEDColor(0.0);
            }

            // Automatic Mode Switching Logic
            if (automatic_mode_switching_enabled && !refractory_period && deviation_history.rows() > 5)
            {
                bool fast_push_detected_this_tick = false;
                bool being_held_detected_this_tick = false;
                bool should_return_to_normal = true;

                if (torque.sum() == 0)
                {
                    should_return_to_normal = false; // If torque is disabled, compliant mode is active
                    // Don't early-return; still update LEDs and state below
                }

                // Compute stats per motor over the last HistoryLength samples
                int rows = deviation_history.rows();
                

                // 1. Detect Fast Push (sharp peak not sustained)
                for (int i = 0; i < deviation.size(); i++) {
                    double lastAbs = std::abs((double)deviation(i));
                    double prevAbs = std::abs((double)deviation.last()[i]);
                    DeviationStats stats(deviation_history, i, allowed_deviance(i), rows);
                    
                    double dev_change = lastAbs - prevAbs;
                    bool sharp_peak = (lastAbs > stats.mean + 2 * stats.stddev) && 
                                    (stats.exceed_ratio <= peak_width_tolerance);

                    if ((dev_change > fast_push_deviation_increase.as_int() || sharp_peak) &&
                        lastAbs > (double)allowed_deviance(i)) {
                        fast_push_detected_this_tick = true;
                        Debug("ForceCheck: Fast push detected for motor " + std::to_string(i) +
                              ", dev_change: " + std::to_string(dev_change) +
                              ", lastAbs: " + std::to_string(lastAbs) +
                              ", mean+2std: " + std::to_string(stats.mean + 2.0 * stats.stddev));
                        break;
                    }
                }

                // 2. Detect Sustained Non-Decreasing Deviation (being held) via elevated mean and minimal movement
                bool sustained_non_decreasing_deviation_this_tick = false;
                for (int i = 0; i < deviation.size(); i++)
                {
                    double lastAbs = std::abs((double)deviation(i));
                    double prevAbs = std::abs((double)deviation.last()[i]);
                    // Check if deviation is not decreasing over last 5 samples
                    bool deviation_not_decreasing = false;
                    if (rows >= 5) {
                        double earliest_abs = std::abs((double)deviation_history(rows-5, i));
                        bool generally_not_decreasing = lastAbs >= earliest_abs * 0.95; // Allow 5% decrease tolerance over 5 samples
                        deviation_not_decreasing = generally_not_decreasing;
                    } else {
                        deviation_not_decreasing = lastAbs >= prevAbs * 0.95; // Fallback to previous comparison if insufficient history
                    }

                    double sum = 0.0; int n = 0;
                    for (int r = 0; r < rows; ++r) { sum += std::abs((double)deviation_history(r, i)); n++; }
                    double meanAbs = n>0 ? sum/n : 0.0;

                    bool high_deviation = meanAbs > (double)allowed_deviance(i) * (double)compliant_trigger_min_deviation_ratio;
                    bool minimal_movement = std::abs((double)present_position(i) - (double)present_position.last()[i]) < (double)max_movement_degrees_when_held.as_int();

                    if (high_deviation && deviation_not_decreasing && minimal_movement)
                    {
                        sustained_non_decreasing_deviation_this_tick = true;
                        Debug("ForceCheck: Sustained non-decreasing deviation candidate motor " + std::to_string(i) +
                              ", meanAbs: " + std::to_string(meanAbs) +
                              ", lastAbs: " + std::to_string(lastAbs));
                        break;
                    }
                }

                if (sustained_non_decreasing_deviation_this_tick)
                {
                    if (!evaluating_sustained_hold)
                    {
                        evaluating_sustained_hold = true;
                        sustained_high_dev_start_time = GetTime();
                        Debug("ForceCheck: Started evaluating sustained non-decreasing deviation.");
                    }
                    else
                    {
                        double hold_duration = (GetTime() - sustained_high_dev_start_time) * 1000.0; // Convert to milliseconds
                        if (hold_duration > sustained_hold_duration_ms.as_int())
                        {
                            being_held_detected_this_tick = true;
                            Debug("ForceCheck: Sustained non-decreasing deviation confirmed. Duration: " + std::to_string(hold_duration) + "ms");
                        }
                    }
                }
                else
                {
                    if (evaluating_sustained_hold)
                    {
                        Debug("ForceCheck: Sustained non-decreasing deviation condition ended.");
                    }
                    evaluating_sustained_hold = false;
                }

                // 3. Assess calm conditions to return to Normal
                bool significant_deviation_present = false;
                bool obstacle_conditions_present = false;
                for (int i = 0; i < deviation.size(); i++)
                {
                    double sum = 0.0; int n = 0; int exceed = 0;
                    for (int r = 0; r < rows; ++r) {
                        double v = std::abs((double)deviation_history(r, i));
                        sum += v; n++; if (v > (double)allowed_deviance(i) * 0.7) exceed++;
                    }
                    double meanAbs = n>0 ? sum/n : 0.0;
                    double exceedRatio = n>0 ? (double)exceed/(double)n : 0.0;
                    if (meanAbs > (double)allowed_deviance(i) * 0.7) {
                        significant_deviation_present = true;
                    }
                    if (exceedRatio > peak_width_tolerance) {
                        obstacle_conditions_present = true;
                    }
                }
                if (significant_deviation_present || obstacle_conditions_present)
                    should_return_to_normal = false;

                // Determine desired mode based on priority and conditions
                int current_mode_idx = mode_controller->GetModeIndex();
                int desired_mode_idx = current_mode_idx;

                if (fast_push_detected_this_tick)
                {
                    desired_mode_idx = 1; // AvoidantMode - highest priority
                    Debug("ForceCheck: Fast push detected - switching to Avoidant mode");
                }
                else if (being_held_detected_this_tick)
                {
                    desired_mode_idx = 2; // CompliantMode - second priority
                    Debug("ForceCheck: Sustained non-decreasing deviation detected - switching to Compliant mode");
                }
                else if (should_return_to_normal && current_mode_idx != 0)
                {
                    desired_mode_idx = 0; // Return to Normal mode when conditions are calm
                    Debug("ForceCheck: Conditions calm - returning to Normal mode");
                }
                else if (current_mode_idx != 0)
                {
                    // Stay in current non-normal mode if there's still significant deviation or obstacles
                    Debug("ForceCheck: Staying in " + std::string(mode_controller->GetCurrentModeName()) +
                          " mode due to ongoing conditions");
                }

                // Apply mode switch if needed
                if (desired_mode_idx != current_mode_idx)
                {
                    Debug("ForceCheck: Auto switching from '" + std::string(mode_controller->GetCurrentModeName()) +
                          "' to '" + GetModeNameByIndex(desired_mode_idx) + "'");
                    control_mode = desired_mode_idx;
                    mode_controller->SwitchMode(desired_mode_idx);
                    last_manual_mode_setting = desired_mode_idx;
                    refractory_period = true;
                    refractory_start_time = GetTime();
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
               
            }
        }
        
        if (present_position.connected() && present_position.size() > 0) {
            previous_position.copy(present_position); // Update for next tick's "held" detection
        }
        firstTick=false;
        //allowed_deviance.print(); // Debug print of allowed deviance

        // Debug prints can be enabled as needed
    }
};

INSTALL_CLASS(ForceCheck);

