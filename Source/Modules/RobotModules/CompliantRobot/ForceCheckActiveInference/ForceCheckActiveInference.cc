/*
 * ForceCheck_active_inference.cc
 * 
 * Enhanced force regulation system integrating:
 * 1. Active Inference for mode selection
 * 2. Variational Free Energy for uncertainty estimation
 * 3. Online learning for precision parameter adaptation
 * 4. Movement-context aware thresholds
 * 
 * Key VFE Features:
 * - Epistemic uncertainty quantification from prediction variance
 * - Adaptive thresholds incorporating movement effort and uncertainty
 * - Online precision parameter updates based on prediction errors
 * - Enhanced external force detection using movement context
 * - Direction-aware force discrimination (opposing vs. assisting movement)
 * 
 * The system distinguishes between:
 * - Large movements with high prediction error (normal operation)
 * - External forces opposing or assisting movement (interaction)
 * - Uncertain predictions (conservative behavior)
 * 
 * Parameters for tuning VFE behavior:
 * - UseUncertaintyEstimation: Enable VFE features
 * - VFELearningRate: Precision adaptation rate (0.001-0.1)
 * - UncertaintyThresholdScale: Uncertainty sensitivity (0.5-5.0) 
 * - PrecisionUpdateRate: Update frequency in seconds (0.5-5.0)
 * - OnlineLearningEnabled: Continuous learning from errors
 */

#include "ikaros.h"

using namespace ikaros;

class ActiveInferenceController
{
private:
    // Matrices for active inference
    matrix precision_weights;    // Precision for each motor based on context
    matrix expected_free_energy; // EFE for each possible action/mode
    matrix epistemic_value;      // Information gain from each action
    matrix pragmatic_value;      // Goal achievement value
    matrix entropy_observations; // Uncertainty in sensory observations
    matrix entropy_states;       // Uncertainty in hidden states
    matrix allowed_deviance;     // Reference to allowed deviance for calculations
    matrix deviation_history; // Stores recent deviations for each motor
    ikaros::Component *parent_component;

    // Model parameters
    double temperature;             // Softmax temperature for action selection
    double precision_learning_rate; // How fast precision adapts
    
    // Temporal dynamics detection
    matrix deviation_velocity;      // Rate of change in deviations
    matrix force_duration_count;    // How long each motor has been experiencing force
    matrix sharp_peak_detected;     // Boolean flags for sharp peaks per motor
    matrix prolonged_force_detected; // Boolean flags for prolonged force per motor
    
    // Adaptive hyperparameters
    struct ModeWeights {
        double risk_weight;
        double ambiguity_weight; 
        double epistemic_weight;
    };
    
    ModeWeights normal_weights;
    ModeWeights avoidant_weights;
    ModeWeights compliant_weights;

public:
    ActiveInferenceController(ikaros::Component *parent) : 
        parent_component(parent),
        temperature(1.0), 
        precision_learning_rate(0.1),
        expected_free_energy(3),          // 3 modes
        epistemic_value(3),               // 3 modes
        pragmatic_value(3),               // 3 modes
        precision_weights(2),             // Will be resized in Initialize()
        entropy_observations(2),          // Will be resized in Initialize()
        entropy_states(2),                // Will be resized in Initialize()
        allowed_deviance(2),              // Will be resized in Initialize()
        deviation_history(10,2),          // Will be resized in Initialize()
        deviation_velocity(2),            // Will be resized in Initialize()
        force_duration_count(2),          // Will be resized in Initialize()
        sharp_peak_detected(2),           // Will be resized in Initialize()
        prolonged_force_detected(2)       // Will be resized in Initialize()
    {
        // Initialize matrices to zero
        expected_free_energy.reset();
        epistemic_value.reset();
        pragmatic_value.reset();
        
        // Initialize default hyperparameters
        normal_weights = {1.0, 0.2, 0.5};      // Balanced
        avoidant_weights = {1.0, 0.2, 0.3};    // Lower epistemic drive 
        compliant_weights = {1.0, 0.1, 0.4};   // REDUCED epistemic drive to prevent over-selection
    }
    
    void Initialize(int num_motors) {
        // Resize matrices to correct size
        precision_weights.resize(num_motors);
        allowed_deviance.resize(num_motors);
        entropy_observations.resize(num_motors);
        entropy_states.resize(num_motors);
        deviation_history.resize(10, num_motors); // 10 time steps, num_motors columns
        
        // Resize temporal detection matrices
        deviation_velocity.resize(num_motors);
        force_duration_count.resize(num_motors);
        sharp_peak_detected.resize(num_motors);
        prolonged_force_detected.resize(num_motors);
        
        precision_weights.set(1.0); // Default precision
        allowed_deviance.reset(); // Initialize to zero (will be set properly later)
        
        // Initialize other matrices
        entropy_observations.reset();
        entropy_states.reset();
        
        // Initialize deviation history as 2D matrix: [time_step, motor_index]
        deviation_history.reset(); // Initialize to zero
        
        // Initialize temporal detection matrices
        deviation_velocity.reset();
        force_duration_count.reset();
        sharp_peak_detected.reset();
        prolonged_force_detected.reset();
    }
    
    void SetAllowedDeviance(const matrix& input_deviance) {
        allowed_deviance.copy(input_deviance);
    }
    
    double CalculateRecentErrorVariance(int motor_index, matrix& deviance_history) {
        // Calculate variance of recent deviations for a motor
        if (deviance_history.size() == 0 ) {
            return 1; // Default fallback
        }
        
        int history_length = deviance_history.rows(); // Number of time steps stored
        if (history_length <= 1) {
            parent_component ->Warning("Not enough history to calculate variance for motor " + std::to_string(motor_index));
            return 1; // Not enough history
        }
        
        double column_sum = 0.0;

        
        // Calculate mean (only count non-zero entries)
        for (int t = 0; t < history_length; t++) {
            double val = deviance_history(t, motor_index);
            if (val != 0.0) { // Only include actual data points
                column_sum += val;

            }
            
        }
        
        if (history_length <= 1) return 1;
        double mean = column_sum/ history_length;
        
        // Calculate variance
        double variance = 0.0;
        for (int t = 0; t < history_length; t++) {
            double val = deviance_history(t, motor_index);
            if (val != 0.0) {
                double diff = val - mean;
                variance += diff * diff;
            }
        }
        
        variance /= history_length; 
        return std::max(0.01, variance); // Ensure minimum variance
    }
    
    matrix& GetExpectedFreeEnergy() { return expected_free_energy; }
    
    // Detect temporal patterns in deviations
    void UpdateTemporalPatterns(matrix &deviation, matrix &number_deviations_per_time_window) {
        for (int i = 0; i < deviation.size(); i++) {
            double current_dev = std::abs(deviation(i));
            double allowed_dev = (i < allowed_deviance.size()) ? allowed_deviance(i) : 1.0;
            double excess_deviation = std::max(0.0, current_dev - allowed_dev);
            
            // Calculate deviation velocity (rate of change)
            if (deviation_history.rows() >= 2) {
                double prev_dev = std::abs(deviation_history(deviation_history.rows()-2, i));
                deviation_velocity(i) = current_dev - prev_dev;
            }
            
            // Sharp peak detection: high deviation velocity AND high current deviation
            double velocity_threshold = allowed_dev * 0.5; // 50% of allowed deviation per tick
            double peak_threshold = allowed_dev * 2.0;     // 200% of allowed deviation
            
            if (std::abs(deviation_velocity(i)) > velocity_threshold && current_dev > peak_threshold) {
                sharp_peak_detected(i) = 1.0;
                parent_component->Debug("Sharp peak detected on motor " + std::to_string(i) + 
                                      " (velocity: " + std::to_string(deviation_velocity(i)) + 
                                      ", deviation: " + std::to_string(current_dev) + ")");
            } else {
                sharp_peak_detected(i) = std::max(0.0, sharp_peak_detected(i) - 0.1); // Decay flag
            }
            
            // Prolonged force detection: sustained high deviation count
            if (number_deviations_per_time_window(i) > 2) {
                force_duration_count(i)++;
                if (force_duration_count(i) > 10) { // 15 ticks of sustained force
                    prolonged_force_detected(i) = 1.0;
                    parent_component->Debug("Prolonged force detected on motor " + std::to_string(i) + 
                                          " (duration: " + std::to_string(force_duration_count(i)) + " ticks)");
                }
            } else {
                force_duration_count(i) = std::max(0.0, force_duration_count(i) - 1.0); // Decay count
                if (force_duration_count(i) <= 5) {
                    prolonged_force_detected(i) = 0.0; // Reset flag when force subsides
                }
            }
        }
    }
    // Calculate precision weights based on motor state and recent prediction errors
    void UpdatePrecisionWeights(matrix &deviation, matrix &motor_in_motion,
                                matrix &number_deviations_per_time_window, matrix &deviation_history)
    {
        for (int i = 0; i < deviation.size(); i++)
        {
            // Core Free Energy Principle: Precision = Inverse Variance
            double recent_error_variance = CalculateRecentErrorVariance(i, deviation_history);
            double base_precision = 1.0 / (recent_error_variance + 0.01); // Add small epsilon to avoid division by zero
            
            // Optional: Context-dependent precision modulation
            // Increse precision when motor is stationary as an anttention mechanism
            if (motor_in_motion[i] ==0)
            {
                base_precision *= 2.0; // Increase precision when stationary
            }
         
            
            // Normalize precision to reasonable range (0.1 to 2.0)
            precision_weights[i] = std::min(2.0, std::max(0.1, base_precision));
            parent_component->Debug("Motor " + std::to_string(i) + " precision: " + std::to_string(precision_weights[i]) +
                                      ", recent error variance: " + std::to_string(recent_error_variance));
        }
    }

    // Calculate expected free energy for each mode
    void CalculateExpectedFreeEnergy(matrix &deviation, matrix &present_position,
                                     matrix &goal_position_in)
    {
        // Mode 0: Normal - minimize prediction error while achieving goals
        expected_free_energy(0) = CalculateEFE_Normal(deviation, present_position, goal_position_in);

        // Mode 1: Retraction - minimize surprise from external forces while considering goals
        expected_free_energy(1) = CalculateEFE_Retraction(deviation, present_position, goal_position_in);

        // Mode 2: Compliant - minimize control effort, maximize information gain while considering goals
        expected_free_energy(2) = CalculateEFE_Compliant(deviation, present_position, goal_position_in);
    }

    double CalculateEFE_Normal(matrix &deviation, matrix &present_position, matrix &goal_position_in)
    {
        double risk = 0.0;              // Preference violation (pragmatic)
        double ambiguity = 0.0;         // Uncertainty in sensory mapping
        double epistemic_value = 0.0;   // Information gain

        for (int i = 0; i < deviation.size(); i++)
        {
            double goal_distance = goal_position_in(i) - present_position(i);
            double prediction_error = deviation(i);
            double precision = precision_weights(i);
            double variance = 1.0 / (precision + 1e-6); // Implied variance
            
            // Calculate excess deviation - only deviations beyond allowed threshold indicate problems
            double allowed_dev = (i < allowed_deviance.size()) ? allowed_deviance(i) : 1.0; // Fallback to 1.0
            double excess_deviation = std::max(0.0, std::abs(prediction_error) - allowed_dev);

            // Risk: squared goal distance weighted by precision (encourages goal reaching)
            // Only penalize excess deviations that indicate actual problems
            risk += precision * (goal_distance * goal_distance + 0.1 * excess_deviation * excess_deviation);

            // Ambiguity: log variance term (uncertainty in observations)
            ambiguity += std::log(variance + 1e-6);

            // Epistemic value: reward high precision (information already gained)
            epistemic_value += precision;
        }

        // Use default weights for normal mode
        return (normal_weights.risk_weight * risk) + (normal_weights.ambiguity_weight * ambiguity) - (normal_weights.epistemic_weight * epistemic_value);
    }

    double CalculateEFE_Retraction(matrix &deviation, matrix &present_position, matrix &goal_position_in)
    {
        double risk = 0.0;              // Preference violation (safety priority)
        double ambiguity = 0.0;         // Uncertainty in sensory mapping
        double epistemic_value = 0.0;   // Information gain

        // Check if sharp peaks are detected - this should favor avoidant mode
        bool sharp_peaks_present = false;
        for (int i = 0; i < deviation.size(); i++) {
            if (sharp_peak_detected(i) > 0.5) {
                sharp_peaks_present = true;
                break;
            }
        }

        for (int i = 0; i < deviation.size(); i++)
        {
            double goal_distance = goal_position_in(i) - present_position(i);
            double prediction_error = deviation(i);
            double precision = precision_weights(i);
            double variance = 1.0 / (precision + 1e-6);
            
            // Calculate excess deviation - only deviations beyond allowed threshold indicate external forces
            double allowed_dev = (i < allowed_deviance.size()) ? allowed_deviance(i) : 1.0; // Fallback to 1.0
            double excess_deviation = std::max(0.0, std::abs(prediction_error) - allowed_dev);
          
            // Risk: Combined safety cost and goal distance cost
            // Safety: heavily penalize excess deviations that indicate external forces/obstacles
            double safety_weight = sharp_peaks_present ? 4.0 : 3.0; // Increase weight for sharp peaks
            double safety_cost = safety_weight * precision * excess_deviation * excess_deviation;
            
            // Goal cost: moderate penalty for being far from goal (safety mode still cares about goals)
            double goal_weight = 0.8; // Reduced compared to normal mode but still present
            double goal_cost = goal_weight * precision * goal_distance * goal_distance;
            
            risk += safety_cost + goal_cost;

            // Ambiguity: log variance term
            ambiguity += std::log(variance + 1e-6);

            // Epistemic value: reduced in retraction mode (safety over exploration)
            epistemic_value += 0.25 * precision;
        }

        // Adaptive weights based on temporal patterns
        ModeWeights weights = avoidant_weights;
        if (sharp_peaks_present) {
            weights.epistemic_weight *= 0.5; // Even lower epistemic drive for sharp peaks
        }

        return (weights.risk_weight * risk) + (weights.ambiguity_weight * ambiguity) - (weights.epistemic_weight * epistemic_value);
    }

    double CalculateEFE_Compliant(matrix &deviation, matrix &present_position, matrix &goal_position_in)
    {
        double risk = 0.0;              // Preference violation (minimal in compliant mode)
        double ambiguity = 0.0;         // Uncertainty in sensory mapping
        double epistemic_value = 0.0;   // Information gain (high in compliant mode)

        // Check if prolonged force is detected - this should favor compliant mode
        bool prolonged_forces_present = false;
        for (int i = 0; i < deviation.size(); i++) {
            if (prolonged_force_detected(i) > 0.5) {
                prolonged_forces_present = true;
                break;
            }
        }

        for (int i = 0; i < deviation.size(); i++)
        {
            double goal_distance = goal_position_in(i) - present_position(i);
            double prediction_error = deviation(i);
            double precision = precision_weights(i);
            double variance = 1.0 / (precision + 1e-6);
            
            // Calculate excess deviation - only deviations beyond allowed threshold matter
            double allowed_dev = (i < allowed_deviance.size()) ? allowed_deviance(i) : 1.0; // Fallback to 1.0
            double excess_deviation = std::max(0.0, std::abs(prediction_error) - allowed_dev);

            // Risk: Combined compliance tolerance and goal distance cost
            // Compliance: minimal cost for excess deviations (tolerates external forces)
            double compliance_weight = 0.8; // Even lower weight for excess deviations
            double compliance_cost = compliance_weight * precision * excess_deviation * excess_deviation;
            
            // Goal cost: still care about reaching goals even in compliant mode
            double goal_weight = 0.6; // Moderate goal pursuit while being compliant
            double goal_cost = goal_weight * precision * goal_distance * goal_distance;
            
            risk += compliance_cost + goal_cost;

            // Ambiguity: Reduced penalty for uncertainty (exploration mode)
            ambiguity += 0.5 * std::log(variance + 1e-6);

            // Epistemic value: Information gain through compliance - but more selective
            double exploration_gain = prolonged_forces_present ? (1.0 + 0.3 * excess_deviation) : 1.0;
            epistemic_value += exploration_gain * precision;
        }

        // Adaptive weights based on temporal patterns
        ModeWeights weights = compliant_weights;
        if (prolonged_forces_present) {
            weights.epistemic_weight = 0.8; // Increase epistemic drive for prolonged forces
            weights.risk_weight = 0.8;      // Reduce risk penalty for prolonged forces
        } else {
            weights.epistemic_weight = 0.2; // Much lower epistemic drive without prolonged force
        }

        return (weights.risk_weight * risk) + (weights.ambiguity_weight * ambiguity) - (weights.epistemic_weight * epistemic_value);
    }

    // Select mode based on minimum expected free energy
    int SelectMode()
    {
        int best_mode = 0;
        double min_efe = expected_free_energy[0];

        for (int i = 1; i < 3; i++)
        {
            if (expected_free_energy[i] < min_efe)
            {
                min_efe = expected_free_energy[i];
                best_mode = i;
            }
        }

        // Add some stochasticity with softmax
        return SoftmaxActionSelection();
    }

    int SoftmaxActionSelection()
    {
        matrix action_probabilities(3);
        double sum_exp = 0.0;

        // Convert EFE to probabilities via softmax
        for (int i = 0; i < 3; i++)
        {
            action_probabilities[i] = exp(-expected_free_energy[i] / temperature);
            sum_exp += action_probabilities[i];
        }

        // Normalize
        for (int i = 0; i < 3; i++)
        {
            action_probabilities[i] /= sum_exp;
        }

        // Sample from distribution (simplified - you might want proper random sampling)
        double rand_val = (double)rand() / RAND_MAX;
        double cumsum = 0.0;
        for (int i = 0; i < 3; i++)
        {
            cumsum += action_probabilities[i];
            if (rand_val < cumsum)
                return i;
        }
        return 2; // Fallback
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
                                matrix& number_deviations_per_time_window, matrix& torque,
                                double time_window, double pullback_amount, double current_time) = 0;
    virtual void SetLEDColor(double deviance_ratio) = 0;
    virtual const char* GetModeName() const = 0;
};

// Normal pose tracking mode
class NormalMode : public ControlMode {
private:
    bool obstacle_detected = false; // Flag to indicate if an obstacle is detected
    double last_obstacle_time;
    matrix halt_goal_position; // Stores the halt goal position
public:
    NormalMode(ikaros::Component* parent, matrix& force_out, matrix& goal_out, matrix& led_int,
               matrix& led_eyes, matrix& led_mouth)
        : ControlMode(parent, force_out, goal_out, led_int, led_eyes, led_mouth) {}

    void HandleDeviation(matrix& deviation, matrix& present_position,
                        matrix& goal_position_in, matrix& start_position,
                        matrix& allowed_deviance, matrix& started_transition,
                        matrix& number_deviations_per_time_window, matrix& torque,
                        double time_window, double pullback_amount, double current_time) override {

        if (present_position.size() == 0 || goal_position_in.size() == 0 || goal_position_out.size() == 0)
        {
            parent_component->Warning("Normal Mode: Empty input matrices (present_position or goal_position_in), skipping HandleDeviation.");
            if (goal_position_out.size() > 0) goal_position_out.reset(); // Still ensure output is reset if possible
            return;
        }
       
        torque.set(1); // Set torque to 1 for normal operation
        
        bool an_obstacle_requires_active_avoidance_this_tick = false;
      
        for (int i = 0; i < deviation.size(); i++)
        {
            if (number_deviations_per_time_window(i) > 4 && !obstacle_detected)
            { // This threshold could be a parameter
                an_obstacle_requires_active_avoidance_this_tick = true;
                halt_goal_position.copy(present_position);

            }
        }

        if (an_obstacle_requires_active_avoidance_this_tick){
            obstacle_detected = true;
            last_obstacle_time = current_time;
            goal_position_out.copy(halt_goal_position); // Set goal position to halt position
            parent_component->Debug("NormalMode: Obstacle detected, halting affected motors.");

            
        } else {
            if (obstacle_detected) { 
                if (current_time - last_obstacle_time > 1.0) { // 1000ms = 1.0s
                    obstacle_detected = false; 
                    // goal_position_out is already goal_position_in (pass-through by default)
                    parent_component->Debug("NormalMode: Obstacle detection timed out, resuming normal goal tracking.");
                    goal_position_out.reset(); // Reset to allow normal goal tracking
                } else {
                    // Still in timeout period, maintain halt if it was set
                    parent_component->Debug("NormalMode: Still in obstacle avoidance timeout, maintaining halt at position " + halt_goal_position.json());
                }
            }
            
        }
        parent_component->Debug("NormalModeL: Present tilt position: " + std::to_string(present_position(0)));
    }
    
    void SetLEDColor(double deviance_ratio) override {
        matrix color(3);
        
        if (obstacle_detected) { // Prioritize red if obstacle is actively being handled
            // Red for high deviation/obstacle
            color(0) = 1.0;
            color(1) = 0.0; 
            color(2) = 0.0;
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
    double last_avoidance_trigger_time;
public:
    AvoidantMode(ikaros::Component* parent, matrix& force_out, matrix& goal_out, matrix& led_int, 
                 matrix& led_eyes, matrix& led_mouth)
        : ControlMode(parent, force_out, goal_out, led_int, led_eyes, led_mouth) {}
    
    void HandleDeviation( matrix& deviation,  matrix& present_position,
                        matrix& goal_position_in,  matrix& start_position,
                        matrix& allowed_deviance,  matrix& started_transition,
                        matrix& number_deviations_per_time_window, matrix& torque,  
                        double time_window, double pullback_amount, double current_time) override {
        if (present_position.size() == 0 || goal_position_in.size() == 0 || goal_position_out.size() == 0) {
            parent_component->Warning("AvoidantMode: Empty input matrices, skipping HandleDeviation.");
            return;
        }
        torque.set(1);

        goal_position_out.reset(); // Initialize goal_position_out to zero
        
        
        bool an_obstacle_requires_active_avoidance_this_tick = false;
        
        
        if (!obstacle_detected) {
            avoidance_goal_position.copy(goal_position_in); // Start with the input goal position
            for (int i = 0; i < deviation.size(); i++)
            {
                // Condition for active avoidance: high deviation count OR if obstacle_detected is already true (meaning a fast push likely triggered this mode)
                if (number_deviations_per_time_window[i] > 2)
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
            last_avoidance_trigger_time = current_time;
            goal_position_out.copy(avoidance_goal_position); 

        } else {
            if (obstacle_detected) { 
                if (current_time - last_avoidance_trigger_time > 1.5) { // 1500ms = 1.5s
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
    double time_of_stabilisation; // Managed internally
    double torque_disabled_time; // Track when torque was disabled
    bool get_stabilisation_time = true; // Managed internally
    bool compliance_mode_active; // True if conditions for compliance were met and torque was disabled
    bool high_deviance_detected_internally = false; // Internal flag for this mode's logic
    matrix last_stable_position; // Used for stabilization check
    matrix previous_position; // Used to track last position for stability checks
    int stable_count = 0; // Instance variable for stability counting

public:
    CompliantMode(ikaros::Component* parent, matrix& force_out, matrix& goal_out, matrix& led_int, 
                  matrix& led_eyes, matrix& led_mouth)
        : ControlMode(parent, force_out, goal_out, led_int, led_eyes, led_mouth), 
          torque_disabled(false), compliance_mode_active(false) {}
    
    void HandleDeviation(matrix& deviation, matrix& present_position,
                        matrix& goal_position_in, matrix& start_position,
                        matrix& allowed_deviance, matrix& started_transition,
                        matrix& number_deviations_per_time_window, matrix& torque,
                        double time_window, double pullback_amount, double current_time) override {
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
            torque_disabled_time = current_time; // Record when torque was disabled
            torque.set(0); // Disable torque
            parent_component->Debug("ForceCheck: CompliantMode - Torque disabled due to high deviance");
        }
        else if (compliance_mode_active && torque_disabled)
        {
            // Check if external force has stopped by monitoring deviations
            bool external_force_present = false;
            for (int i = 0; i < deviation.size(); i++)
            {
                if (number_deviations_per_time_window[i] > 1) // Lower threshold to detect ongoing external force
                {
                    external_force_present = true;
                    break;
                }
            }
            
            // If external force is still present, reset stability counting
            if (external_force_present)
            {
                stable_count = 0;
                get_stabilisation_time = true;
                parent_component->Debug("ForceCheck: CompliantMode - External force still detected, maintaining compliance");
                return; // Keep torque disabled while force is present
            }
            
            // External force has stopped, now check if position has stabilized to re-enable torque
            int position_margin = 2; // Margin for position stability

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
                    time_of_stabilisation = current_time;
                    get_stabilisation_time = false; // Set to false to avoid resetting time
                }

                parent_component->Debug("ForceCheck: Compliance motor stable duration: " + std::to_string((current_time - time_of_stabilisation) * 1000.0) + "ms");
                if (current_time - time_of_stabilisation > 1.2) // 1200ms = 1.2s
                {
                    parent_component->Debug("ForceCheck: Compliance motor re-enabling torque after stabilization and delay.");
                    torque.set(1);
                    torque_disabled = false;
                    high_deviance_detected = false; // Reset high deviance flag
                    stable_count = 0; // Reset stable count
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
                        double time_window, double pullback_amount, double current_time) {
        if (!current_mode) {
            parent_component->Error("ModeController: current_mode is null in HandleDeviation!");
            SwitchMode(0); // Attempt to recover by switching to NormalMode
        }
        current_mode->HandleDeviation(deviation, present_position, goal_position_in,
                                    start_position, allowed_deviance, started_transition,
                                    number_deviations_per_time_window, torque, time_window, pullback_amount, current_time);
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

class ForceCheckActiveInference: public Module
{
public: // Ensure INSTALL_CLASS can access constructor if it's implicitly used.
    //parameters
    parameter pullback_amount;
    parameter time_window;
    parameter control_mode; 
    
    // New parameters for automatic mode switching
    parameter automatic_mode_switching_enabled;
    
    // Active Inference parameters
    parameter temperature;
    parameter precision_learning_rate;

    
    // Variational Free Energy parameters
    parameter use_uncertainty_estimation;
    parameter vfe_learning_rate;
    parameter uncertainty_threshold_scale;
    parameter precision_update_rate;
    parameter online_learning_enabled;

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
    matrix deviation_history; // Stores recent deviations for each motor
    bool firstTick;

    // Variational Free Energy and uncertainty estimation
    matrix epistemic_uncertainty;      // Uncertainty from prediction variance
    matrix aleatoric_uncertainty;      // Inherent model uncertainty
    matrix movement_context_features;  // Enhanced movement context
    matrix adaptive_threshold_base;    // VFE-computed thresholds
    matrix precision_weights;          // Learned precision parameters
    matrix expected_free_energy_output; // Expected free energy for each mode (for output)
    
    // Online learning buffers
    matrix prediction_history;         // Recent predictions
    matrix observation_history;        // Recent actual currents
    matrix error_history;             // Recent prediction errors
    int history_buffer_size;
    int current_history_index;
    double last_precision_update_time;

    bool goal_reached;
    double start_time;
    int goal_time_out;
    bool force_output_tapped;
    double force_output_tapped_time_point;
    double time_window_start;

    // Variables for auto mode switching
    bool refractory_period = false;                      // Avoid rapid mode switching

    

    // Add mode controller
    std::unique_ptr<ModeController> mode_controller;
    
    // Add active inference controller
    std::unique_ptr<ActiveInferenceController> active_inference_controller;

   
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
        Bind(time_window, "TimeWindow");
        Bind(led_intensity, "LedIntensity");
        Bind(led_color_eyes, "LedColorEyes");
        Bind(led_color_mouth, "LedColorMouth");
    
        Bind(torque, "Torque");
        Bind(control_mode, "ControlMode"); 
        
        // VFE and Active Inference Outputs
        Bind(epistemic_uncertainty, "EpistemicUncertainty");
        Bind(precision_weights, "PrecisionWeights");
        Bind(adaptive_threshold_base, "AdaptiveThresholds");
        Bind(expected_free_energy_output, "ExpectedFreeEnergy");
        Bind(movement_context_features, "MovementContext");
        
        // Mode switching parameters
        Bind(automatic_mode_switching_enabled, "AutomaticModeSwitchingEnabled");
        
        // Active Inference parameters
        Bind(temperature, "Temperature");
        Bind(precision_learning_rate, "PrecisionLearningRate");

        
        // Variational Free Energy parameters
        Bind(use_uncertainty_estimation, "UseUncertaintyEstimation");
        Bind(vfe_learning_rate, "VFELearningRate");
        Bind(uncertainty_threshold_scale, "UncertaintyThresholdScale");
        Bind(precision_update_rate, "PrecisionUpdateRate");
        Bind(online_learning_enabled, "OnlineLearningEnabled");


        torque.set(1); // Enable torque by default

        start_time = std::time(nullptr); // Using C time, consider chrono if more precision needed elsewhere
        goal_time_out = 5; // Seconds
        firstTick = true;
        // obstacle = false; // Handled by mode logic or new auto-switch logic
        // obstacle_time = 0; // Handled by mode logic
        goal_reached = false;
        previous_position.set_name("PreviousPosition");
  
        position_margin = 1; // Degrees
        led_intensity.set(0.5); // Default intensity
        led_color_eyes.set(1); // Default white
        led_color_mouth.set(1); // Default white

        red_color_RGB = {1,0.3,0.3}; // Example, not directly used if SetLEDColor in modes handles all
        force_output_tapped = false; // Unused?
        force_output_tapped_time_point = GetTime(); // Unused?

        number_deviations_per_time_window.set_name("NumberDeviationsPerTimeWindow");
        number_deviations_per_time_window.copy(deviation); // Ensure size matches present_position
        number_deviations_per_time_window.reset(); // Initialize to zero

        time_window_start = GetTime();
        allowed_deviance.copy(allowed_deviance_input); // Initialize allowed_deviance with input

        deviation_history= matrix(1, deviation.size()); // [motor_index, time_step] - store last 10 for each motor
                           // Initialize to zero
        deviation_history.set_name("DeviationHistory");
        // Note: deviation_history will be properly initialized in first tick when deviation size is known
        
        // Initialize mode controller
        mode_controller = std::make_unique<ModeController>(this, force_output, goal_position_out, 
                                                          led_intensity, led_color_eyes, led_color_mouth);
        
        // Initialize active inference controller
        active_inference_controller = std::make_unique<ActiveInferenceController>(this);

        // Initialize deviation history matrix to store last 10 deviations for each motor
        // Will be properly sized when deviation matrix is available in first tick
        
        // Initialize VFE variables
        history_buffer_size = 50; // Store last 50 predictions/observations
        current_history_index = 0;
        last_precision_update_time = 0.0; // Will be set to GetTime() in first tick

        prediction_history.set_name("PredictionHistory");
        observation_history.set_name("ObservationHistory");
        error_history.set_name("ErrorHistory");

        prediction_history = matrix(0, deviation.size());
        observation_history = matrix(0, deviation.size());
        error_history = matrix(0, deviation.size());
    }
    
    const char* GetModeNameByIndex(int index) {
        if (index == 0) return "Normal";
        if (index == 1) return "Avoidant";
        if (index == 2) return "Compliant";
        return "Unknown";
    }

    // Variational Free Energy Methods
    void ComputeEnhancedMovementContext(matrix& present_position, 
                                      matrix& goal_position_in, 
                                      matrix& start_position) {
        if (!use_uncertainty_estimation) return;
        
        if (movement_context_features.size() != deviation.size() * 8) {
            movement_context_features = matrix(deviation.size() * 8);
        }
        
        for (int i = 0; i < deviation.size(); i++) {
            int base_idx = i * 8;
            
            // Feature 1: Movement velocity (from position history)
            double velocity = 0.0;
            if (!firstTick && previous_position.size() > i) {
                velocity = abs(present_position(i) - previous_position(i));
            }
            movement_context_features(base_idx) = velocity;
            
            // Feature 2: Distance to goal (normalized)
            double dist_to_goal = abs(goal_position_in(i) - present_position(i)) / 180.0;
            movement_context_features(base_idx + 1) = dist_to_goal;
            
            // Feature 3: Distance from start (normalized)  
            double dist_from_start = abs(present_position(i) - start_position(i)) / 180.0;
            movement_context_features(base_idx + 2) = dist_from_start;
            
            // Feature 4: Movement direction
            double movement_direction = (goal_position_in(i) > present_position(i)) ? 1.0 : -1.0;
            movement_context_features(base_idx + 3) = movement_direction;
            
            // Feature 5: Position load factor (simplified model)
            double position_load = abs(present_position(i)) / 90.0;
            movement_context_features(base_idx + 4) = position_load;
            
            // Feature 6: Movement effort (distance * load)
            double movement_effort = dist_to_goal * position_load;
            movement_context_features(base_idx + 5) = movement_effort;
            
            // Feature 7: Movement phase (0=start, 1=end)
            double total_movement = abs(goal_position_in(i) - start_position(i));
            double movement_phase = (total_movement > 0) ? 
                abs(present_position(i) - start_position(i)) / total_movement : 0.0;
            movement_context_features(base_idx + 6) = movement_phase;
            
            // Feature 8: Acceleration placeholder
            movement_context_features(base_idx + 7) = 0.0;
        }
    }
    
    void UpdatePredictionObservationHistory(matrix& deviation) {
        if (!online_learning_enabled) return;
        
        
        
        
        
        int idx = current_history_index % history_buffer_size;
        
        
        prediction_history.push(current_prediction, true);
        observation_history.push(present_current, true);
        error_history.push(deviation, true);
        
        current_history_index++;
    }
    
    void ComputeEpistemicUncertainty() {
        if (!use_uncertainty_estimation) return;
        
        if (epistemic_uncertainty.size() != deviation.size()) {
            epistemic_uncertainty = matrix(deviation.size());
            aleatoric_uncertainty = matrix(deviation.size());
        }
        
        int window_size = std::min(10, prediction_history.rows());
        
        for (int i = 0; i < deviation.size(); i++) {
            if (window_size < 3) {
                epistemic_uncertainty(i) = 0.0;
                continue;
            }
            
            double mean_pred = 0.0;
            for (int j = 0; j < window_size; j++) {
                int idx = (current_history_index - 1 - j + history_buffer_size) % history_buffer_size;
                if (idx < prediction_history.rows()) {
                    mean_pred += prediction_history(idx, i);
                }
            }
            mean_pred /= window_size;
            
            double var_pred = 0.0;
            for (int j = 0; j < window_size; j++) {
                int idx = (current_history_index - 1 - j + history_buffer_size) % history_buffer_size;
                if (idx < prediction_history.rows()) {
                    double diff = prediction_history(idx, i) - mean_pred;
                    var_pred += diff * diff;
                }
            }
            var_pred /= (window_size - 1);
            
            epistemic_uncertainty(i) = sqrt(var_pred);
        }
    }
    
    void UpdatePrecisionParameters() {
        if (!use_uncertainty_estimation) return;
        
        if (precision_weights.size() != deviation.size()) {
            precision_weights = matrix(deviation.size());
            precision_weights.set(1.0);
        }
        
        int window_size = std::min(20, error_history.rows());
        if (window_size < 5) return;
        
        for (int i = 0; i < deviation.size(); i++) {
            double mean_error = 0.0;
            for (int j = 0; j < window_size; j++) {
                int idx = (current_history_index - 1 - j + history_buffer_size) % history_buffer_size;
                if (idx < error_history.rows()) {
                    mean_error += error_history(idx, i);
                }
            }
            mean_error /= window_size;
            
            double var_error = 0.0;
            for (int j = 0; j < window_size; j++) {
                int idx = (current_history_index - 1 - j + history_buffer_size) % history_buffer_size;
                if (idx < error_history.rows()) {
                    double diff = error_history(idx, i) - mean_error;
                    var_error += diff * diff;
                }
            }
            var_error /= (window_size - 1);
            
            double target_precision = 1.0 / (1.0 + var_error);
            double alpha = vfe_learning_rate;
            precision_weights(i) = (1.0 - alpha) * precision_weights(i) + alpha * target_precision;
        }
    }
    
    void ComputeVFEAdaptiveThresholds() {
        if (!use_uncertainty_estimation) return;
        
        if (adaptive_threshold_base.size() != deviation.size()) {
            adaptive_threshold_base = matrix(deviation.size());
        }
        
        for (int i = 0; i < deviation.size(); i++) {
            double base = allowed_deviance_input.size() > i ? allowed_deviance_input(i) : 50.0;
            
            double movement_scale = 1.0;
            if (movement_context_features.size() > i * 8 + 5) {
                int feature_base = i * 8;
                double movement_effort = movement_context_features(feature_base + 5);
                double movement_phase = movement_context_features(feature_base + 6);
                
                movement_scale = 1.0 + movement_effort * 3.0;
                double phase_scale = 1.0 + movement_phase * (1.0 - movement_phase) * 2.0;
                movement_scale *= phase_scale;
            }
            
            double uncertainty_scale = 1.0;
            if (epistemic_uncertainty.size() > i) {
                uncertainty_scale = 1.0 + epistemic_uncertainty(i) * uncertainty_threshold_scale;
            }
            
            double precision_scale = precision_weights.size() > i ? precision_weights(i) : 1.0;
            precision_scale = std::max(0.1, precision_scale);
            
            adaptive_threshold_base(i) = base * movement_scale * uncertainty_scale / precision_scale;
        }
        
        for (int i = 0; i < allowed_deviance.size() && i < adaptive_threshold_base.size(); i++) {
            allowed_deviance(i) = adaptive_threshold_base(i);
        }
    }
    
    bool IsExternalForceVFE(int motor_idx) {
        if (!use_uncertainty_estimation || motor_idx >= deviation.size()) {
            return false;
        }
        
        double current_dev_abs = abs(deviation(motor_idx));
        double adaptive_threshold = adaptive_threshold_base.size() > motor_idx ? 
            adaptive_threshold_base(motor_idx) : allowed_deviance_input(motor_idx);
        
        bool exceeds_adaptive_threshold = current_dev_abs > adaptive_threshold;
        
        bool opposes_movement = false;
        if (movement_context_features.size() > motor_idx * 8 + 3) {
            int feature_base = motor_idx * 8;
            double movement_direction = movement_context_features(feature_base + 3);
            double deviation_direction = (deviation(motor_idx) > 0) ? 1.0 : -1.0;
            opposes_movement = (movement_direction * deviation_direction < 0) && 
                              (motor_in_motion.size() > motor_idx && motor_in_motion[motor_idx] == 1);
        }
        
        bool high_uncertainty = false;
        if (epistemic_uncertainty.size() > motor_idx) {
            double uncertainty_threshold = 0.2 * adaptive_threshold;
            high_uncertainty = epistemic_uncertainty(motor_idx) > uncertainty_threshold;
        }
        
        bool low_precision = false;
        if (precision_weights.size() > motor_idx) {
            low_precision = precision_weights(motor_idx) < 0.5;
        }
        
        return exceeds_adaptive_threshold && (opposes_movement || high_uncertainty || low_precision);
    }



    void Tick()
    {
        deviation.copy(current_prediction);
        deviation.subtract(present_current);

        if (deviation_history.rows() == 10)
        {                              // Every 10 ticks, reset
            deviation_history.resize(0, 2);
        }

        if (present_current.size() == 0 || present_position.size() == 0 || goal_position_in.size() == 0 ||
            start_position.size() == 0 || allowed_deviance.size() == 0) {
            Warning("ForceCheck: One or more input matrices are empty, skipping Tick logic.");
            return; // Skip Tick logic if inputs are not ready
        }
        if (firstTick) {
            previous_position.copy(present_position); // Initialize previous_position on first tick
            
           
            // Initialize active inference controller with correct motor count
            if (deviation.size() > 0) {
                active_inference_controller->Initialize(deviation.size());
                active_inference_controller->SetAllowedDeviance(allowed_deviance);
                
                // Initialize VFE matrices with correct sizes
                if (use_uncertainty_estimation) {
                    epistemic_uncertainty = matrix(deviation.size());
                    aleatoric_uncertainty = matrix(deviation.size());
                    movement_context_features = matrix(deviation.size() * 8);
                    adaptive_threshold_base = matrix(deviation.size());
                    precision_weights = matrix(deviation.size());
                    expected_free_energy_output = matrix(3); // 3 modes: Normal, Avoidant, Compliant
                    
                    epistemic_uncertainty.set(0.0);
                    aleatoric_uncertainty.set(0.0);
                    precision_weights.set(1.0);
                    expected_free_energy_output.set(0.0);
                    
                    last_precision_update_time = GetTime();
                    
                    Debug("VFE: Initialized matrices for " + std::to_string(deviation.size()) + " motors");
                }
            }
            
            firstTick = false; // Set to false after first initialization
        }
        if (deviation_history.size() >0)
        {   
            deviation_history.push(deviation, true); // Store current deviation in history
            deviation_history.print(); // Debug print of deviation history
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
        if (GetTime() - time_window_start > time_window.as_int())
        {
            time_window_start = GetTime();
            if (number_deviations_per_time_window.size() > 0) number_deviations_per_time_window.reset();
            Debug("ForceCheck: Time window exceeded, resetting deviation count.");
        }

        //Check refractory period for automatic mode switching
        if (GetTime() - refractory_start_time > 2.0) { // 2 seconds
            Debug("ForceCheck: Refractory period ended, allowing mode switching again.");
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

            // Variational Free Energy Processing
            if (use_uncertainty_estimation) {
                // 1. Compute enhanced movement context features
                ComputeEnhancedMovementContext(present_position, goal_position_in, start_position);
                
                // 2. Update prediction/observation history for online learning
                UpdatePredictionObservationHistory(deviation);
                
                // 3. Compute epistemic uncertainty from prediction variance
                ComputeEpistemicUncertainty();
                
                // 4. Update precision parameters periodically
                if ((GetTime() - last_precision_update_time) > precision_update_rate) {
                    UpdatePrecisionParameters();
                    last_precision_update_time = GetTime();
                }
                
                // 5. Compute adaptive thresholds incorporating uncertainty and movement context
                ComputeVFEAdaptiveThresholds();
                
                Debug("VFE: Updated - epistemic_uncertainty: " + epistemic_uncertainty.json() + 
                      ", precision_weights: " + precision_weights.json());
            }

            // Update number_deviations_per_time_window
            for (int i = 0; i < allowed_deviance_input.size(); i++) {
                
                if (abs(deviation[i]) > allowed_deviance_input(i)) {
                    number_deviations_per_time_window[i]++;
                }
                Debug("ForceCheck: Motor " + std::to_string(i) + " deviation: " + std::to_string(deviation[i]) +
                      ", dev within TW: " + std::to_string(number_deviations_per_time_window[i]) +
                      ", moving: " + std::to_string(motor_in_motion[i]));
            }
            
            mode_controller->HandleDeviation(deviation, present_position, goal_position_in,
                                                start_position, 
                                                use_uncertainty_estimation ? adaptive_threshold_base : allowed_deviance_input, 
                                                started_transition, // started_transition might be empty
                                                number_deviations_per_time_window, torque, (double)time_window,
                                                (double)pullback_amount, GetTime());
            if (allowed_deviance_input.size() > 0 && allowed_deviance_input[0] > 0)
            {                                      // Avoid div by zero
                mode_controller->SetLEDColor(0.0); // No real deviation, so 0 ratio
            }
            else
            {
                mode_controller->SetLEDColor(0.0);
            }

            // Use VFE-enhanced thresholds for active inference controller
            active_inference_controller->SetAllowedDeviance(
                use_uncertainty_estimation ? adaptive_threshold_base : allowed_deviance_input);
            active_inference_controller->UpdatePrecisionWeights(deviation, motor_in_motion,
                                                                number_deviations_per_time_window, deviation_history);
            
            // Update temporal patterns BEFORE calculating EFE
            active_inference_controller->UpdateTemporalPatterns(deviation, number_deviations_per_time_window);
            
            active_inference_controller->CalculateExpectedFreeEnergy(deviation, present_position,
                                                                     goal_position_in);

            // Update the output matrix with expected free energy values
            if (use_uncertainty_estimation && expected_free_energy_output.size() == 3) {
                matrix& efe = active_inference_controller->GetExpectedFreeEnergy();
                if (efe.size() >= 3) {
                    expected_free_energy_output.copy(efe);
                }
            }

            if (automatic_mode_switching_enabled && !refractory_period)
            {
                
                
                if (torque.sum() > 0)
                {
                    

                    int desired_mode_idx = active_inference_controller->SelectMode();
                        int current_mode_idx = mode_controller->GetModeIndex();

                    if (desired_mode_idx != current_mode_idx)
                    {
                        matrix& efe = active_inference_controller->GetExpectedFreeEnergy();
                        Debug("ForceCheck: Active inference switching from '" +
                            std::string(mode_controller->GetCurrentModeName()) +
                            "' to '" + GetModeNameByIndex(desired_mode_idx) +
                            "' (EFE: " + efe.json() + ")");

                        control_mode = desired_mode_idx;
                        mode_controller->SwitchMode(desired_mode_idx);
                        last_manual_mode_setting = desired_mode_idx;
                        refractory_period = true;
                        refractory_start_time = GetTime();
                    }
                    else
                    {
                        // Debug EFE values even when not switching
                        matrix& efe = active_inference_controller->GetExpectedFreeEnergy();
                        Debug("ForceCheck: Mode selection - Current: " + std::string(mode_controller->GetCurrentModeName()) + 
                              " (EFE: " + efe.json() + ")");
                    }
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
            if (allowed_deviance_input.size() > highest_deviance_index && allowed_deviance_input[highest_deviance_index] > 0)
            {
                deviance_ratio = highest_deviance_abs / allowed_deviance_input[highest_deviance_index];
            }
            else if (allowed_deviance_input.size() > 0 && allowed_deviance_input[0] > 0)
            { // Fallback if index is bad but allowed_deviance exists
                deviance_ratio = highest_deviance_abs / allowed_deviance_input[0];
            }

            // // Use the mode controller to handle the deviation
            // mode_controller->HandleDeviation(deviation, present_position, goal_position_in,
            //                                 start_position, allowed_deviance, started_transition,
            //                                 number_deviations_per_time_window, torque, (double)time_window,
            //                                 (double)pullback_amount);

            // // Set LED colors based on current mode and deviance
            // double intensity = clip(deviance_ratio * 0.5 + 0.5, 0.5, 1.0); // Scale ratio so 0 -> 0.5 intensity, 1 -> 1.0 intensity
            // led_intensity.set(intensity);
            // mode_controller->SetLEDColor(deviance_ratio);
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
                if (!allowed_deviance_input.connected() || allowed_deviance_input.size() == 0)
                    Warning("ForceCheck: AllowedDeviance not connected or empty.");
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

INSTALL_CLASS(ForceCheckActiveInference);

