#include "ikaros.h"
#include <chrono>
#include <algorithm>
#include <cmath>

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
    matrix previous_current_prediction; // Store prediction from tick t to compare with current at tick t+
    matrix started_transition;
    matrix motor_in_motion;
    matrix allowed_deviance;
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

    

    double refractory_start_time;

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

        // Pattern indicators
        bool has_sharp_peak;        // Fast push: narrow, high peak
        bool has_sustained_plateau; // Sustained: wide, moderate plateau
        bool has_oscillations;      // Obstacle: variable errors

        int sharp_peak_count; // Number of significant peaks
    };

    enum class RobotState
    {
        NORMAL,     // Following goal normally
        RETRACTING, // Pulling back from push
        COMPLIANT,  // Torque off, free movement
        HALTED      // Stopped at obstacle
    };

    std::vector<ErrorProfile> motor_profiles; // One profile per motor
    RobotState robot_state;

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

    ErrorProfile GetErrorProfile(const matrix& deviation_window, int motor_idx, double threshold = 0)
    {
        ErrorProfile profile = {};
        if (deviation_window.size() == 0 || deviation_window.rows() == 0) return profile;
        
        // Check if motor_idx is valid
        if (motor_idx >= deviation_window.cols()) {
            Warning("ForceCheck: motor_idx out of bounds in GetErrorProfile");
            return profile;
        }
        
        int N = deviation_window.rows();
        
        // Basic statistics for the specified motor
        double sum = 0.0;
        double sum2 = 0.0;
        double max_val = 0.0;
        int max_idx_row = 0;
        int count_above_threshold = 0;
        
        for (int row = 0; row < N; row++) {
            double val = (double)deviation_window(row, motor_idx); // Signed error for this motor
            double abs_val = std::abs(val);
            sum += val;
            sum2 += val * val;
            
            if (abs_val > max_val) {
                max_val = abs_val;
                max_idx_row = row;
            }
            if (val > threshold) count_above_threshold++;
        }
        
        // Mean and standard deviation for this motor
        profile.mean_error = sum / N;
        double variance = (N > 1) ? (sum2 / N - profile.mean_error * profile.mean_error) : 0.0;
        profile.std_dev = std::sqrt(std::max(0.0, variance));
        profile.peak_magnitude = max_val;
        profile.samples_above_threshold = count_above_threshold;
        
        // Peak width (FWHM - samples above half peak magnitude)
        double half_peak = max_val * 0.5;
        int peak_width_samples = 0;
        int search_window = std::min(10, N / 4); // Search ±10 samples or 25% of window
        int search_start = std::max(0, max_idx_row - search_window);
        int search_end = std::min(N, max_idx_row + search_window);
        for (int row = search_start; row < search_end; row++) {
            if (std::abs((double)deviation_window(row, motor_idx)) >= half_peak) {
                peak_width_samples++;
            }
        }
        profile.peak_width = (double)peak_width_samples / N; // As fraction of window
        
        // Slope for this motor
        if (N >= 3) {
            double start_avg = 0.0;
            double end_avg = 0.0;
            int slope_samples = std::min(3, N / 10); // Use 3 samples or 10% of window
            for (int i = 0; i < slope_samples; i++) {
                start_avg += (double)deviation_window(i, motor_idx);
                end_avg += (double)deviation_window(N - slope_samples + i, motor_idx);
            }
            start_avg /= slope_samples;
            end_avg /= slope_samples;
            profile.slope = (end_avg - start_avg) / N; // Change per sample
        } else {
            profile.slope = 0.0;
        }
        
        // Duration (using GetTicksPerSecond for actual timing)
        double tick_duration_ms = 1000.0 / GetTickDuration();
        profile.duration_ms = N * tick_duration_ms;
        
        // Pattern indicators (using class parameters)
        // Sharp peak: high magnitude, narrow width
        profile.has_sharp_peak = (profile.peak_magnitude > peak_height * profile.std_dev) && 
                                 (profile.peak_width < peak_width);
        
        // Sustained plateau: high mean, low variance, many samples above threshold
        double frac_above = (double)count_above_threshold / N;
        double plateau_threshold = threshold * sustained_force_threshold_ratio;
        profile.has_sustained_plateau = (profile.mean_error > plateau_threshold) &&
                                        (frac_above > 0.7) &&
                                        (profile.std_dev < profile.mean_error * 1.5);
        
        // Oscillations: sign changes + moderate variance
        int sign_changes = 0;
        for (int row = 1; row < N; row++) {
            double curr = (double)deviation_window(row, motor_idx);
            double prev = (double)deviation_window(row - 1, motor_idx);
            if ((curr > threshold && prev < -threshold) || (curr < -threshold && prev > threshold)) {
                sign_changes++;
            }
        }
        double oscillation_rate = (double)sign_changes / N;
        profile.has_oscillations = (oscillation_rate > 0.15) && 
                                   (profile.std_dev > profile.mean_error * 0.5);
        
        // Count sharp peaks in window
        profile.sharp_peak_count = 0;
        for (int row = 1; row < N - 1; row++) {
            double curr = std::abs((double)deviation_window(row, motor_idx));
            double prev = std::abs((double)deviation_window(row - 1, motor_idx));
            double next = std::abs((double)deviation_window(row + 1, motor_idx));
            if (curr > threshold && curr > prev && curr > next) {
                profile.sharp_peak_count++;
            }
        }
        
        return profile;
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

        void Init()
        {
            Bind(present_current, "PresentCurrent");
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
            Bind(peak_width, "PeakWidth"); // Width of the peak for sharp peak detection in percentage of deviance history
            Bind(peak_height, "PeakHeight");                  // Height of the peak for sharp peak detection in standard deviations above mean

            // Mode switching parameters
            Bind(automatic_mode_switching_enabled, "AutomaticModeSwitchingEnabled");
            Bind(sustained_hold_duration_ms, "SustainedHoldDurationMs");
            Bind(sustained_force_threshold_ratio, "SustainedForceThresholdRatio");       // Threshold for detecting sustained guidance forces
            Bind(movement_threshold_scalar, "MovementThresholdScalar");                  // Scalar for movement threshold

            torque.set(1); // Enable torque by default

            start_time = std::time(nullptr); // Using C time, consider chrono if more precision needed elsewhere
            goal_time_out = 5;               // Seconds
            goal_reached = false;
            previous_position.set_name("PreviousPosition");
            previous_current_prediction.set_name("PreviousCurrentPrediction");

            led_intensity.set(0.5); // Default intensity
            led_color_eyes.set(1);  // Default white
            led_color_mouth.set(1); // Default white

            // Removed number_deviations_per_time_window init and time_window_start since we're using deviation_history

            deviation_history = matrix(0, deviation.size()); // Store recent deviations for each motor
            deviation_history.set_name("DeviationHistory");

            evaluating_sustained_hold = false;
            allowed_deviance.copy(allowed_deviance_input); // Initialize allowed_deviance with input
            
            
        }

        void Tick()
        {
            
            UpdateDeviationHistory();
            
            // Analyze error profile for each motor
            if (deviation_history.rows() > 0 && deviation_history.cols() > 0) {
                motor_profiles.clear();
                motor_profiles.resize(deviation_history.cols());
                
                for (int motor = 0; motor < deviation_history.cols(); motor++) {
                    motor_profiles[motor] = GetErrorProfile(deviation_history, motor);
                }
            }
            

            // Print error profile for debugging (for each motor)
            for (int motor = 0; motor < motor_profiles.size(); motor++) {
                std::cout << "Motor " << motor << " Error Profile - "
                          << "Mean: " << motor_profiles[motor].mean_error
                          << ", StdDev: " << motor_profiles[motor].std_dev
                          << ", PeakMag: " << motor_profiles[motor].peak_magnitude
                          << ", PeakWidth: " << motor_profiles[motor].peak_width
                          << ", Slope: " << motor_profiles[motor].slope
                          << ", Duration(ms): " << motor_profiles[motor].duration_ms
                          << ", SamplesAboveThreshold: " << motor_profiles[motor].samples_above_threshold
                          << ", SharpPeak: " << motor_profiles[motor].has_sharp_peak
                          << ", SustainedPlateau: " << motor_profiles[motor].has_sustained_plateau
                          << ", Oscillations: " << motor_profiles[motor].has_oscillations
                          << ", SharpPeakCount: " << motor_profiles[motor].sharp_peak_count
                          << std::endl;
            }

            // robot_state = UpdateRobotState(error_profile, robot_state);

            // ExecuteBehaviour(robot_state);
        }
    };

INSTALL_CLASS(ForceCheck);

