#include "ikaros.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace ikaros;


    

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
    parameter max_movement_degrees_when_held;
    parameter obstacle_detection_deviation_count;
    parameter sustained_force_threshold_ratio; // Threshold for detecting sustained guidance forces (lower than normal threshold)
    parameter movement_threshold_scalar;
    parameter peak_width; // Width of the peak for sharp peak detection in percentage of devaiance history
    parameter peak_height;
    parameter confidence_threshold;

    
    // Classification threshold parameters
    parameter push_min_magnitude;
    parameter push_max_magnitude;
    parameter obstacle_min_std_dev;
    parameter obstacle_min_peaks;
    parameter refactory_duration;

    //inputs
    matrix present_current;
    matrix current_limit;
    matrix present_position;// assumes degrees
    matrix goal_position_in;
    matrix start_position;
    matrix allowed_deviance_input;
    matrix perturbation_classification; // New input for perturbation classification

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
    matrix previous_current_prediction; // Store prediction from tick t to compare with current at tick t+
    matrix started_transition;
    matrix motor_in_motion;
    matrix allowed_deviance;
    matrix deviation_history; // Stores recent deviations for each motor
    matrix colour; // RBG


    bool goal_reached;
    double start_time;
    int goal_time_out;
    bool force_output_tapped;
    double force_output_tapped_time_point;
    // Removed time_window_start (no time-window logic anymore)

    // Variables for auto mode switching
    double sustained_high_dev_start_time;
    bool refractory_period;                      // Avoid rapid mode switching
    bool fast_push_detected;                     // True if a fast push was detected
    bool evaluating_sustained_hold;              // True if currently evaluating sustained hold

    bool all_motors_stable;
    bool use_ml_classifier; // True if all motors are stable

    double refractory_start_time;
    double motors_stable_start_time; // Time when motors first became stable (for compliant state)

    struct ErrorProfile
    {
        // Statistical features
        double mean_error;     // Average over window
        double std_dev;        // Variability
        double peak_magnitude; // Max absolute error
        double peak_width;     // Duration of peak
        double slope;          // Rising/falling rate
        

        // Temporal features
        double duration_ms;          // How long has pattern lasted
        int samples_above_threshold; // Count of significant errors
        double elevation_start_time; // Time when error first became elevated (for sustained detection)
        double sustained_duration;   // Actual measured duration of sustained elevation

        // Pattern indicators
        bool has_sharp_peak;        // Fast push: narrow, high peak
        bool has_sustained_plateau; // Sustained: wide, moderate plateau
        bool has_oscillations;      // Obstacle: variable errors

        int sharp_peak_count; // Number of significant peaks
        std::string pattern_type;   // Classified pattern type
        double confidence;          // Confidence score [0.0-1.0] for the classification
    };

    enum class RobotState
    {
        NORMAL,     // Following goal normally
        PUSH, // Pulling back from push
        COMPLIANT,  // Torque off, free movement
        OBSTACLE      // Stopped at obstacle
    };

    struct MotorState
    {
        RobotState state;          // Individual state for this motor
        double state_switch_time;  // Time when current state was entered (last state switch)
    };

    std::vector<MotorState> motor_states; // One state per motor (includes individual RobotState)
    
    // Track sustained plateau timing per motor (persistent across ticks)
   matrix elevation_start_times; // When each motor first became elevated
   matrix is_currently_elevated; // Whether each motor is currently elevated

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
        matrix transition_status(deviation.size());
        if (present_position.size() == 0 || present_position.size() != start_position.size())
        {
            Warning("ForceCheck: StartedTransition - Input matrix size mismatch or empty. Returning empty matrix.");
            return transition_status;
        }

         
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
      
        for (int i = 0; i < deviation.size(); i++) {
            bool is_moving = false;
            
            // Method 1: Check if motor has moved since last tick (more sensitive threshold)
            try {
                matrix previous_position = present_position.last();
                if (previous_position.size() == present_position.size()) {
                    double position_change = abs((double)present_position(i) - (double)previous_position(i));
                    if (position_change > 0.5) { // 0.5 degrees threshold for movement detection
                        is_moving = true;
                    }
                }
            } catch (...) {
                // No history available
            }
            
            
            
            // Method 2: Check current magnitude (motor working hard = likely moving)
            if (present_current.size() > i && abs((double)present_current(i)) > 15) {
                is_moving = true;
            }
            
            in_motion(i) = is_moving ? 1.0 : 0.0;
        }
        
        return in_motion;
    }

    void UpdateDeviationHistory()
    {
        if (current_prediction.connected())
        {
            deviation.copy(current_prediction);
            deviation.subtract(present_current);
        }
        else
        {
            Error("ForceCheck: CurrentPrediction input not connected.");
            return;
        }

        // Only resize if necessary and ensure safe bounds
        if (deviation.size() > 0 &&
            (deviation_history.cols() != deviation.size() || deviation_history.rows() > history_lenght.as_int()))
        {
            deviation_history.realloc(0, deviation.size()); // Use realloc for matrix
            // Reinitialize cache after resize - ensure proper size
        }

        try
        {
            deviation_history.push(deviation, true);
        }
        catch (const std::exception &e)
        {
            // If push fails due to buffer full, reset and try again
            deviation_history.realloc(0, deviation.size()); // Use realloc for matrix

            deviation_history.push(deviation, true);
        } // Append current deviation to history

       
    }

    void ApplyColorToAllLEDs(const matrix &color)
    {
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

    void Init()
    {
            Bind(present_current, "PresentCurrent");
            Bind(present_position, "PresentPosition");
            Bind(goal_position_in, "GoalPositionIn");
            Bind(perturbation_classification, "PerturbationClassification");
            use_ml_classifier = perturbation_classification.connected();

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
            Bind(peak_width, "PeakWidth");
            Bind(peak_height, "PeakHeight");
            Bind(confidence_threshold, "MinConfidenceForStateSwitch");
            Bind(automatic_mode_switching_enabled, "AutomaticModeSwitchingEnabled");
            Bind(sustained_hold_duration_ms, "SustainedHoldDurationMs");
            Bind(sustained_force_threshold_ratio, "SustainedForceThresholdRatio");
            Bind(movement_threshold_scalar, "MovementThresholdScalar");
            Bind(refactory_duration, "RefactoryDuration");
           
            
            
         
        
            goal_position_out.set(0);

            colour = matrix(3); // RGB

            led_intensity.set(0.5);

           
            led_color_eyes.set(1.0);

            led_color_mouth.set(1.0);

       
            torque.set(1);

            motor_states.resize(current_prediction.size());
            for(auto& state : motor_states) {
                state.state = RobotState::NORMAL;
                state.state_switch_time = 0.0;
            }
            
            refractory_period = false;
            refractory_start_time = 0.0;
            motors_stable_start_time = 0.0;
        }

        
       
        void Tick()
        {
            UpdateDeviationHistory();

            if (!use_ml_classifier || present_position.size() == 0 || perturbation_classification.size() < 1) {
                return;
            }

            
            RobotState current_state = motor_states[0].state;

            // Check for motor stabilization first
            bool all_motors_stabilized = true;
            previous_position = present_position.last();
            if (previous_position.size() == present_position.size()) {
                for (int i = 0; i < present_position.size(); ++i) {
                    if (abs(present_position(i) - previous_position.last()(i)) > 1.0) { // 1 degree threshold for stability
                        all_motors_stabilized = false;
                        break;
                    }
                }
            } else {
                all_motors_stabilized = false;
            }
            double current_time = GetTime();
            // Track how long motors have been stable (for COMPLIANT state)
            if (all_motors_stabilized) {
                if (motors_stable_start_time == 0.0) {
                    motors_stable_start_time = current_time;
                }
            } else {
                motors_stable_start_time = 0.0; // Reset if motors start moving
            }

            // If in COMPLIANT state, stay there until motors are stabilized for 2 seconds
            if (current_state == RobotState::COMPLIANT) {
                if (motors_stable_start_time > 0.0 && (current_time - motors_stable_start_time >= 2.0)) {
                    // Motors have been stable for 2 seconds, return to NORMAL
                    for (auto& motor_state : motor_states) {
                        motor_state.state = RobotState::NORMAL;
                        motor_state.state_switch_time = current_time;
                    }
                    motors_stable_start_time = 0.0; // Reset
                    // No refractory period when returning to normal
                }
                // Skip classifier processing while in COMPLIANT state
                // (will stay in COMPLIANT until stabilized)
            } else {
                // Refractory period check - only applies when leaving PUSH or OBSTACLE
                if (refractory_period && (current_time - refractory_start_time < (double)refactory_duration)) {
                    // While in refractory, just maintain current state's outputs
                    // But handle automatic state transitions
                } else {
                    refractory_period = false;

                    int class_idx = (int)perturbation_classification(0);
                    RobotState new_state = current_state;

                    // Determine new state from classifier ONLY if confidence is high enough
                    if (perturbation_classification(1) >= (double)confidence_threshold) {
                        // Confidence exceeds threshold - trust the classification
                        switch (class_idx) {
                            case 1: new_state = RobotState::OBSTACLE; break;
                            case 2: new_state = RobotState::PUSH; break;
                            case 3: new_state = RobotState::COMPLIANT; break;
                            default: new_state = RobotState::NORMAL; break;
                        }
                    } else {
                        // Low confidence - stay in current state
                        new_state = current_state;
                    }

                    // State transition logic
                    if (new_state != current_state) {
                        // Only apply refractory period when transitioning TO push/obstacle/compliant, not FROM them
                        bool should_apply_refractory = (new_state == RobotState::PUSH || 
                                                       new_state == RobotState::OBSTACLE || 
                                                       new_state == RobotState::COMPLIANT);
                        
                        for (auto& motor_state : motor_states) {
                            motor_state.state = new_state;
                            motor_state.state_switch_time = current_time;
                        }
                        
                        if (should_apply_refractory) {
                            refractory_period = true;
                            refractory_start_time = current_time;
                        }
                        
                        // Reset motor stability tracking when entering COMPLIANT state
                        if (new_state == RobotState::COMPLIANT) {
                            motors_stable_start_time = 0.0;
                        }
                    }
                }

                // After refractory period, any perturbation state (PUSH, OBSTACLE) should return to NORMAL
                // (COMPLIANT is handled separately above based on motor stabilization)
                if ((current_state == RobotState::PUSH || current_state == RobotState::OBSTACLE) && 
                    (current_time - motor_states[0].state_switch_time >= (double)refactory_duration)) {
                    for (auto& motor_state : motor_states) {
                        motor_state.state = RobotState::NORMAL;
                        motor_state.state_switch_time = current_time;
                    }
                    // No refractory period when returning to normal
                }
            }

            // Default outputs
            goal_position_out.set(0);
            torque.set(1);
            
            perturbation_classification.print();

            // State-based actions (apply outputs based on current state)
            switch (motor_states[0].state) {
                    case RobotState::NORMAL:
                        colour(0) = 1.0; colour(1) = 1.0; colour(2) = 1.0; // White
                        
                        break;

                    case RobotState::OBSTACLE:
                        goal_position_out.copy(present_position);
                        colour(0) = 1.0; colour(1) = 0.0; colour(2) = 0.0; // Red
                        
                        led_intensity(0) = 0.8;
                        break;

                    case RobotState::PUSH:
                        goal_position_out.copy(present_position);
                        for (int i = 0; i < current_prediction.size(); ++i) {
                            double retraction = (goal_position_in.size() > i && goal_position_in(i) > 180) ? -pullback_amount : pullback_amount;
                            goal_position_out(i) = present_position(i) + retraction;
                        }
                        colour(0) = 1.0; colour(1) = 1.0; colour(2) = 0.0; // Yellow
                        led_intensity(0) = 0.8;
                        break;

                    case RobotState::COMPLIANT:
                        torque.set(0);
                        // State transition logic is handled above - just maintain compliant behavior
                        colour(0) = 0.0; colour(1) = 1.0; colour(2) = 0.0; // Green
                        
                        led_intensity(0) = 0.8;
                        break;
                }

                ApplyColorToAllLEDs(colour);
        }
    };

INSTALL_CLASS(ForceCheck);

