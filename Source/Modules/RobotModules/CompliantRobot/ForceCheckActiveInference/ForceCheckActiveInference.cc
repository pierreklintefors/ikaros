/*
 * ForceCheck_active_inference.cc
 * 
 * Active Inference system for policy selection based on minimizing prediction error:
 * 
 * POLICIES (Actions):
 * 0. CONTINUE: Keep moving toward goal (normal operation)
 * 1. STOP: Halt movement and maintain current position
 * 2. REFLECT: Pull back from external force (avoidance)
 * 3. COMPLY: Disable torque and let external force move the robot
 * 
 * The system selects the policy that minimizes Expected Free Energy (EFE):
 * EFE = Risk (pragmatic value) + Ambiguity (epistemic value)
 * 
 * Risk: Expected deviation from preferred states (low current prediction error)
 * Ambiguity: Uncertainty about sensory outcomes (information gain potential)
 * 
 * Each policy is evaluated by predicting its effect on future prediction errors
 * and selecting the one that minimizes long-term surprise.
 */

#include "ikaros.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>


using namespace ikaros;

// Helper struct for deviation statistics
struct DeviationStats {
    double mean = 0.0;
    double stddev = 0.0;
    double exceed_ratio = 0.0;
    int exceed_count = 0;
    int total_samples = 0;
    
    DeviationStats(const matrix& deviation_history, int motor_idx, double threshold, int rows = -1) {
        if (deviation_history.rows() == 0 || motor_idx >= deviation_history.cols()) return;
        
        // Use actual matrix rows if rows parameter is -1 or invalid
        int actual_rows = (rows == -1 || rows > deviation_history.rows()) ? deviation_history.rows() : rows;
        
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

// Forward declaration
class ForceCheckActiveInference;

class ActiveInferenceController
{
private:
    // Matrices for active inference
    matrix precision_weights;    // Precision for each motor based on context
    matrix expected_free_energy; // EFE for each possible policy
    matrix predicted_errors;     // Predicted current errors for each policy
    matrix policy_priors;        // Prior preferences for each policy
    matrix allowed_deviance;     // Reference to allowed deviance for calculations
    
    // Feature detection matrices for Active Inference
    matrix deviation_velocity;          // Rate of change in deviation
    matrix sharp_peak_detected;         // Flag for sharp force peaks
    matrix force_duration_count;        // Duration counter for sustained forces
    matrix prolonged_force_detected;    // Flag for prolonged forces
    
    ikaros::Component *parent_component;

    // Model parameters
    double temperature;             // Softmax temperature for policy selection
    double precision_learning_rate; // How fast precision adapts
    
    // Mode weight configurations for Active Inference
    struct ModeWeights {
        double risk_weight;
        double ambiguity_weight; 
        double epistemic_weight;
    };
    
    ModeWeights normal_weights = {1.0, 0.5, 0.2};     // Normal operation
    ModeWeights avoidant_weights = {2.0, 1.0, 0.8};   // Higher sensitivity
    ModeWeights compliant_weights = {0.3, 0.2, 0.1};  // Lower resistance
    
    // Prediction models for each policy
    struct PolicyModel {
        double prediction_weight;   // How much this policy affects predictions
        double error_reduction;     // Expected error reduction from this policy
        double uncertainty_cost;    // Cost of uncertainty under this policy
    };
    
    PolicyModel continue_model;   // Policy 0: Continue movement
    PolicyModel stop_model;       // Policy 1: Stop movement  
    PolicyModel reflect_model;    // Policy 2: Reflect/pullback
    PolicyModel comply_model;     // Policy 3: Comply/disable torque

    // Helper method for calculating recent error variance
    double CalculateRecentErrorVariance(int motor_idx, const matrix& history, int count, int head) {
        if (count < 2 || motor_idx >= history.cols()) return 0.0;
        
        double mean = 0.0;
        int samples = std::min(count, 10); // Use last 10 samples max
        
        // Calculate mean
        for (int i = 0; i < samples; i++) {
            int idx = (head - 1 - i + history.rows()) % history.rows();
            mean += history(idx, motor_idx);
        }
        mean /= samples;
        
        // Calculate variance
        double variance = 0.0;
        for (int i = 0; i < samples; i++) {
            int idx = (head - 1 - i + history.rows()) % history.rows();
            double diff = history(idx, motor_idx) - mean;
            variance += diff * diff;
        }
        variance /= samples;
        
        return variance;
    }

public:
    ActiveInferenceController(ikaros::Component *parent) : 
        parent_component(parent),
        temperature(1.0), 
        precision_learning_rate(0.1),
        expected_free_energy(4),          // 4 policies
        predicted_errors(4),              // 4 policies
        policy_priors(4),                 // 4 policies
        precision_weights(2),             // Will be reinitialized in Initialize()
        allowed_deviance(2)               // Will be reinitialized in Initialize()
    {
        // Initialize matrices to zero
        expected_free_energy.reset();
        predicted_errors.reset();
        
        // Initialize policy priors (slight preference for continuing)
        policy_priors(0) = 0.6;  // CONTINUE - slight preference
        policy_priors(1) = 0.15; // STOP - neutral
        policy_priors(2) = 0.15; // REFLECT - neutral  
        policy_priors(3) = 0.1;  // COMPLY - slight penalty (only when really needed)
        
        // Initialize policy models
        continue_model = {1.0, 0.1, 0.2};   // Normal prediction, slight error reduction, low uncertainty cost
        stop_model = {0.8, 0.3, 0.4};       // Reduced prediction, better error reduction, medium uncertainty cost
        reflect_model = {0.9, 0.5, 0.3};    // Good prediction, good error reduction, medium uncertainty cost
        comply_model = {0.3, 0.8, 0.1};     // Poor prediction, high error reduction, low uncertainty cost
    }
    
    void Initialize(int num_motors) {
        // Initialize matrices to correct size
        precision_weights = matrix(num_motors);
        allowed_deviance = matrix(num_motors);
        predicted_errors = matrix(4); // 4 policies
        expected_free_energy = matrix(4); // 4 policies
        
        // Initialize Active Inference feature detection matrices
        deviation_velocity = matrix(num_motors);
        sharp_peak_detected = matrix(num_motors);
        force_duration_count = matrix(num_motors);
        prolonged_force_detected = matrix(num_motors);
        
        precision_weights.set(1.0); // Default precision
        allowed_deviance.reset(); // Initialize to zero (will be set properly later)
        predicted_errors.reset();
        expected_free_energy.reset();
        
        // Initialize feature detection matrices
        deviation_velocity.reset();
        sharp_peak_detected.reset();
        force_duration_count.reset();
        prolonged_force_detected.reset();
    }
    
    void SetAllowedDeviance(const matrix& input_deviance) {
        allowed_deviance.copy(input_deviance);
    }
    
    
    matrix& GetExpectedFreeEnergy() { return expected_free_energy; }
    
    // Main method to calculate Expected Free Energy for all policies
    void CalculateExpectedFreeEnergy(matrix &current_deviation, matrix &present_position, matrix &goal_position_in) {
        // Policy 0: CONTINUE - keep moving toward goal
        expected_free_energy(0) = CalculateEFE_Continue(current_deviation, present_position, goal_position_in);
        
        // Policy 1: STOP - halt movement at current position
        expected_free_energy(1) = CalculateEFE_Stop(current_deviation, present_position, goal_position_in);
        
        // Policy 2: REFLECT - pull back from external force
        expected_free_energy(2) = CalculateEFE_Reflect(current_deviation, present_position, goal_position_in);
        
        // Policy 3: COMPLY - disable torque and follow external force
        expected_free_energy(3) = CalculateEFE_Comply(current_deviation, present_position, goal_position_in);
    }
    
    // Calculate EFE for CONTINUE policy (normal movement toward goal)
    double CalculateEFE_Continue(matrix &deviation, matrix &present_position, matrix &goal_position_in) {
        double total_efe = 0.0;
        
        for (int i = 0; i < deviation.size(); i++) {
            double current_error = std::abs(deviation(i));
            double goal_distance = std::abs(goal_position_in(i) - present_position(i));
            double allowed_dev = (i < allowed_deviance.size()) ? allowed_deviance(i) : 50.0;
            double precision = precision_weights(i);
            
            // Predict future error if we continue moving
            // If current error is high and we continue, error may persist or increase
            double predicted_error = current_error * continue_model.prediction_weight;
            
            // If we're moving toward goal normally, error should decrease
            if (current_error <= allowed_dev * 1.2) {
                predicted_error *= (1.0 - continue_model.error_reduction);
            }
            
            // Use excess deviation beyond allowed threshold
            double excess = std::max(0.0, predicted_error - allowed_dev);
            double risk = precision * excess * excess;
            
            // Goal achievement bonus: Prefer continuing if close to goal
            double goal_bonus = -0.5 * precision / (1.0 + goal_distance);
            
            // Uncertainty cost: Lower for continuing (we understand normal movement)
            double uncertainty = continue_model.uncertainty_cost * std::log(1.0 + current_error);
            
            total_efe += risk + uncertainty + goal_bonus;
        }
        
        return total_efe - std::log(policy_priors(0)); // Add policy prior
    }
    
    // Calculate EFE for STOP policy (halt movement)
    double CalculateEFE_Stop(matrix &deviation, matrix &present_position, matrix &goal_position_in) {
        double total_efe = 0.0;
        
        for (int i = 0; i < deviation.size(); i++) {
            double current_error = std::abs(deviation(i));
            double goal_distance = std::abs(goal_position_in(i) - present_position(i));
            double allowed_dev = (i < allowed_deviance.size()) ? allowed_deviance(i) : 50.0;
            double precision = precision_weights(i);
            
            // Predict future error if we stop
            // Stopping should reduce prediction error from movement but may not solve external forces
            double predicted_error = current_error * stop_model.prediction_weight;
            
            // If current error is from movement, stopping helps
            if (current_error > allowed_dev) {
                predicted_error *= (1.0 - stop_model.error_reduction);
            }
            
            // Risk: penalize only excess beyond allowed threshold
            double excess = std::max(0.0, predicted_error - allowed_dev);
            double risk = precision * excess * excess;
            
            // Goal cost: Penalty for not progressing toward goal
            double goal_cost = 0.3 * precision * goal_distance;
            
            // Uncertainty cost: Medium for stopping (we understand stopping but lose info about dynamics)
            double uncertainty = stop_model.uncertainty_cost * std::log(1.0 + current_error);
            
            total_efe += risk + goal_cost + uncertainty;
        }
        
        return total_efe - std::log(policy_priors(1)); // Add policy prior
    }
    
    // Calculate EFE for REFLECT policy (pull back/avoid)
    double CalculateEFE_Reflect(matrix &deviation, matrix &present_position, matrix &goal_position_in) {
        double total_efe = 0.0;
        
        for (int i = 0; i < deviation.size(); i++) {
            double current_error = std::abs(deviation(i));
            double goal_distance = std::abs(goal_position_in(i) - present_position(i));
            double allowed_dev = (i < allowed_deviance.size()) ? allowed_deviance(i) : 50.0;
            double precision = precision_weights(i);
            
            // Predict future error if we reflect/pull back
            // Reflecting should reduce error from external forces but may increase goal distance
            double predicted_error = current_error * reflect_model.prediction_weight;
            
            // If current error is from external force opposing movement, reflecting helps significantly
            if (current_error > allowed_dev * 1.5) {
                predicted_error *= (1.0 - reflect_model.error_reduction);
            }
            
            // Risk: penalize only excess beyond allowed threshold
            double excess = std::max(0.0, predicted_error - allowed_dev);
            double risk = precision * excess * excess;
            
            // Goal cost: Moderate penalty for moving away from goal
            double goal_cost = 0.4 * precision * goal_distance;
            
            // Safety bonus: Reward avoiding potentially harmful external forces
            double safety_bonus = -0.2 * precision * std::max(0.0, current_error - allowed_dev);
            
            // Uncertainty cost: Medium (we learn about external forces but create new uncertainties)
            double uncertainty = reflect_model.uncertainty_cost * std::log(1.0 + current_error);
            
            total_efe += risk + goal_cost + uncertainty + safety_bonus;
        }
        
        return total_efe - std::log(policy_priors(2)); // Add policy prior
    }
    
    // Calculate EFE for COMPLY policy (disable torque)
    double CalculateEFE_Comply(matrix &deviation, matrix &present_position, matrix &goal_position_in) {
        double total_efe = 0.0;
        
        for (int i = 0; i < deviation.size(); i++) {
            double current_error = std::abs(deviation(i));
            double goal_distance = std::abs(goal_position_in(i) - present_position(i));
            double allowed_dev = (i < allowed_deviance.size()) ? allowed_deviance(i) : 50.0;
            double precision = precision_weights(i);
            
            // Predict future error if we comply (disable torque)
            // Complying eliminates prediction error from our actions but external agent controls position
            double predicted_error = current_error * comply_model.prediction_weight;
            
            // If there's external force, complying dramatically reduces prediction error
            if (current_error > allowed_dev ) {
                predicted_error *= (1.0 - comply_model.error_reduction);
            }
            
            // Risk: penalize only excess beyond allowed threshold
            double excess = std::max(0.0, predicted_error - allowed_dev);
            double risk = precision * excess * excess;
            
            // Goal cost: High penalty for giving up goal control
            double goal_cost = 0.8 * precision * goal_distance;
            
            // Information bonus: We learn a lot about external forces by complying
            double info_bonus = -0.3 * precision * std::max(0.0, current_error - allowed_dev);
            
            // Uncertainty cost: Low (we accept uncertainty by not trying to control)
            double uncertainty = comply_model.uncertainty_cost * std::log(1.0 + current_error);
            
            total_efe += risk + goal_cost + uncertainty + info_bonus;
        }
        
        return total_efe - std::log(policy_priors(3)); // Add policy prior
    }
    
    // Detect temporal patterns in deviations
    void UpdateTemporalPatterns(matrix &deviation, matrix &deviation_history, matrix &torque_enabled,
                                int history_count, int history_head) {
        for (int i = 0; i < deviation.size(); i++) {
            double current_dev = std::abs(deviation(i));
            double allowed_dev = (i < allowed_deviance.size()) ? allowed_deviance(i) : 1.0;
            double excess_deviation = std::max(0.0, current_dev - allowed_dev);
            
            // Calculate deviation velocity (rate of change)
            if (history_count >= 2 && deviation_history.cols() > i) {
                int cap = deviation_history.rows();
                int last_idx = (history_head - 1 + cap) % cap;
                int prev_idx = (history_head - 2 + cap) % cap;
                // We already have current_dev from deviation(i); use prev from history
                double prev_dev = std::abs((double)deviation_history(prev_idx, i));
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
            
            // Prolonged force detection: sustained high deviation using deviation history statistics
            if (history_count > 0 && torque_enabled.sum() > 0) {
                DeviationStats stats(deviation_history, i, allowed_dev);
                if (stats.exceed_ratio > 0.3) { // 30% of recent history exceeds threshold
                    force_duration_count(i)++;
                    if (force_duration_count(i) > 10) { // 10 ticks of sustained force
                        prolonged_force_detected(i) = 1.0;
                        parent_component->Debug("Prolonged force detected on motor " + std::to_string(i) + 
                                              " (duration: " + std::to_string(force_duration_count(i)) + " ticks, exceed_ratio: " + 
                                              std::to_string(stats.exceed_ratio) + ")");
                    }
                } else {
                    force_duration_count(i) = std::max(0.0, force_duration_count(i) - 1.0); // Decay count
                    if (force_duration_count(i) <= 5) {
                        prolonged_force_detected(i) = 0.0; // Reset flag when force subsides
                    }
                }
            }
        }
    }
    // Calculate precision weights based on motor state and recent prediction errors
    void UpdatePrecisionWeights(matrix &deviation, matrix &motor_in_motion,
                                matrix &number_deviations_per_time_window, matrix &deviation_history,
                                int history_count, int history_head)
    {
        for (int i = 0; i < deviation.size(); i++)
        {
            // Core Free Energy Principle: Precision = Inverse Variance
            double recent_error_variance = CalculateRecentErrorVariance(i, deviation_history, history_count, history_head);
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

    // Select policy based on minimum expected free energy
    int SelectMode()
    {
        int best_policy = 0;
        double min_efe = expected_free_energy[0];

        for (int i = 1; i < 4; i++)  // Now we have 4 policies
        {
            if (expected_free_energy[i] < min_efe)
            {
                min_efe = expected_free_energy[i];
                best_policy = i;
            }
        }

        // Add some stochasticity with softmax for exploration
        return SoftmaxPolicySelection();
    }

    int SoftmaxPolicySelection()
    {
        matrix policy_probabilities(4);  // 4 policies
        double sum_exp = 0.0;

        // Convert EFE to probabilities via softmax (lower EFE = higher probability)
        for (int i = 0; i < 4; i++)
        {
            policy_probabilities[i] = exp(-expected_free_energy[i] / temperature);
            sum_exp += policy_probabilities[i];
        }

        // Normalize
        for (int i = 0; i < 4; i++)
        {
            policy_probabilities[i] /= sum_exp;
        }

        // Sample from distribution
        double rand_val = (double)rand() / RAND_MAX;
        double cumsum = 0.0;
        for (int i = 0; i < 4; i++)
        {
            cumsum += policy_probabilities[i];
            if (rand_val < cumsum)
                return i;
        }
        return 0; // Fallback to CONTINUE policy
    }
    
    // Update precision weights based on prediction errors
    void UpdatePrecisionWeights(matrix &deviation) {
        for (int i = 0; i < precision_weights.size(); i++) {
            double current_error = std::abs(deviation(i));
            double allowed_dev = (i < allowed_deviance.size()) ? allowed_deviance(i) : 50.0;
            
            // Lower precision when errors are consistently high
            double error_ratio = current_error / allowed_dev;
            double target_precision = 1.0 / (1.0 + error_ratio);
            
            // Smooth update
            precision_weights(i) = (1.0 - precision_learning_rate) * precision_weights(i) +
                                  precision_learning_rate * target_precision;
            
            // Keep precision in reasonable bounds
            precision_weights(i) = std::max(0.1, std::min((double)precision_weights(i), 2.0));
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
                
                // Get previous deviation safely - only check sharp peak if previous data exists
                bool has_previous = false;
                double prevAbs = lastAbs; // Default to current value
                try {
                    matrix prev_dev = deviation.last();
                    if (prev_dev.size() > i) {
                        prevAbs = std::abs((double)prev_dev[i]);
                        has_previous = true;
                    }
                } catch (...) {
                    has_previous = false;
                }
                
                DeviationStats stats(deviation_history, i, allowed_deviance(i), rows);
                
                // Sharp peak detection - only if we have valid previous data
                bool sharp_peak = false;
                if (has_previous) {
                    sharp_peak = (lastAbs > stats.mean + 2 * stats.stddev) && 
                               (prevAbs <= (double)allowed_deviance(i)) && 
                               (stats.exceed_ratio <= peak_width_tolerance);
                }
                
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
        
        // Initialize previous_position if not yet done
        if (previous_position.size() == 0) {
            previous_position = matrix(present_position.size());
            previous_position.copy(present_position);
        }

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
            static int stable_count = 0; // Make this static so it persists across function calls

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
        
        // Update previous_position for next tick's comparison (at the end)
        previous_position.copy(present_position);
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

// Policy Controller - implements the 4 Active Inference policies
class PolicyController {
private:
    ikaros::Component* parent_component;
    matrix& force_output;
    matrix& goal_position_out;
    matrix& torque;
    matrix& led_intensity;
    matrix& led_color_eyes;
    matrix& led_color_mouth;
    
    int current_policy;
    matrix halt_position;
    matrix reflect_position;
    double last_policy_change_time;
    
public:
    PolicyController(ikaros::Component* parent, matrix& force_out, matrix& goal_out, matrix& torque_ref,
                    matrix& led_int, matrix& led_eyes, matrix& led_mouth)
        : parent_component(parent), force_output(force_out), goal_position_out(goal_out), 
          torque(torque_ref), led_intensity(led_int), led_color_eyes(led_eyes), led_color_mouth(led_mouth),
          current_policy(0), last_policy_change_time(0.0) {}

    void ExecutePolicy(int policy_index, matrix& deviation, matrix& present_position,
                      matrix& goal_position_in, double pullback_amount, double current_time) {
        
        if (policy_index != current_policy) {
            parent_component->Debug("PolicyController: Switching from policy " + std::to_string(current_policy) + 
                                   " to policy " + std::to_string(policy_index) + " (" + GetPolicyName(policy_index) + ")");
            current_policy = policy_index;
            last_policy_change_time = current_time;
        }
        
        switch (policy_index) {
            case 0: ExecuteContinue(deviation, present_position, goal_position_in); break;
            case 1: ExecuteStop(deviation, present_position, goal_position_in); break;
            case 2: ExecuteReflect(deviation, present_position, goal_position_in, pullback_amount); break;
            case 3: ExecuteComply(deviation, present_position, goal_position_in); break;
            default: 
                parent_component->Warning("PolicyController: Unknown policy " + std::to_string(policy_index));
                ExecuteContinue(deviation, present_position, goal_position_in);
                break;
        }
        
        UpdateLEDs(policy_index, deviation);
    }
    
private:
    // Policy 0: CONTINUE - normal movement toward goal
    void ExecuteContinue(matrix& deviation, matrix& present_position, matrix& goal_position_in) {
        torque.set(1); // Enable torque
        goal_position_out.reset(); // Let GoalSetter handle goal setting
        parent_component->Debug("PolicyController: Executing CONTINUE - torque enabled, normal goal tracking");
    }
    
    // Policy 1: STOP - halt movement at current position
    void ExecuteStop(matrix& deviation, matrix& present_position, matrix& goal_position_in) {
        torque.set(1); // Keep torque enabled to maintain position
        
        // Set goal to current position to stop movement
        if (halt_position.size() != present_position.size()) {
            halt_position = matrix(present_position.size());
        }
        halt_position.copy(present_position);
        goal_position_out.copy(halt_position);
        parent_component->Debug("PolicyController: Executing STOP - torque enabled, holding position " + halt_position.json());
    }
    
    // Policy 2: REFLECT - pull back from external force
    void ExecuteReflect(matrix& deviation, matrix& present_position, matrix& goal_position_in, double pullback_amount) {
        torque.set(1); // Keep torque enabled for active movement
        
       
        
        for (int i = 0; i < deviation.size(); i++) {
            // Pull back in opposite direction of force (deviation)
            double pullback_direction = (deviation(i) > 0.0) ? -pullback_amount : pullback_amount;
            goal_position_out(i) = present_position(i) + pullback_direction;
        }
        
        
        parent_component->Debug("PolicyController: Executing REFLECT - torque enabled, pulling back to " + reflect_position.json());
    }
    
    // Policy 3: COMPLY - disable torque and follow external force
    void ExecuteComply(matrix& deviation, matrix& present_position, matrix& goal_position_in) {
        torque.set(0); // Disable torque to allow external control
        goal_position_out.reset(); // Don't set any goal - just follow external force
        parent_component->Debug("PolicyController: Executing COMPLY - torque DISABLED, following external force");
    }
    
    void UpdateLEDs(int policy_index, matrix& deviation) {
        matrix color(3);
        
        switch (policy_index) {
            case 0: // CONTINUE - White (normal operation)
                color.set(1.0);
                break;
            case 1: // STOP - Red (stopped/halted)
                color(0) = 1.0; color(1) = 0.0; color(2) = 0.0;
                break;
            case 2: // REFLECT - Yellow (retraction/avoidance)
                color(0) = 1.0; color(1) = 1.0; color(2) = 0.0;
                break;
            case 3: // COMPLY - Green (torque disabled)
                color(0) = 0.0; color(1) = 1.0; color(2) = 0.0;
                break;
        }
        
        // Apply color to all LEDs
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
        
        // Set intensity based on deviation magnitude
        double max_deviation = 0.0;
        for (int i = 0; i < deviation.size(); i++) {
            max_deviation = std::max(max_deviation, std::abs((double)deviation(i)));
        }
        double intensity = std::min(1.0, 0.5 + max_deviation / 100.0);
        led_intensity.set(intensity);
        
        // Debug output for LED status
        parent_component->Debug("PolicyController: LEDs set to " + std::string(GetPolicyName(policy_index)) + 
                               " color (R:" + std::to_string(color(0)) + 
                               " G:" + std::to_string(color(1)) + 
                               " B:" + std::to_string(color(2)) + 
                               ") intensity:" + std::to_string(intensity));
    }
    
public:
    const char* GetPolicyName(int policy_index) const {
        switch (policy_index) {
            case 0: return "CONTINUE";
            case 1: return "STOP";
            case 2: return "REFLECT";
            case 3: return "COMPLY";
            default: return "UNKNOWN";
        }
    }
    
    int GetCurrentPolicy() const { return current_policy; }
    
    void SetCurrentPolicy(int policy_index) { 
        current_policy = policy_index; 
    }
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

class ForceCheckActiveInference: public Module
{
public: // Ensure INSTALL_CLASS can access constructor if it's implicitly used.
    
    // Destructor to save calibration data
    ~ForceCheckActiveInference() {
        if (is_calibrating && calibration_sample_count > 0) {
            SaveCalibrationData();
            Debug("ForceCheck: Calibration completed - saved " + std::to_string(calibration_sample_count) + " samples");
        }
    }
    
    //parameters
    parameter pullback_amount;
    parameter peak_width_tolerance; // Width of the peak for sharp peak detection in percentage of deviance history
    parameter time_window;
    parameter control_mode;
    parameter deviation_history_length;

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
    
    // Calibration parameters
    parameter calibration_mode;


    //inputs
    matrix present_current;
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
    int history_head = 0;     // Points to next write position in circular buffer
    int history_count = 0;    // Current number of valid entries in buffer
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
    double time_window_start;

    // Variables for auto mode switching
    bool refractory_period = false;                      // Avoid rapid mode switching
    
    // Calibration variables
    bool is_calibrating = false;
    double calibration_start_time;
    int calibration_sample_count;
    matrix calibration_precision_accumulator;  // Accumulate precision values during calibration
    matrix calibration_threshold_accumulator;  // Accumulate adaptive thresholds during calibration
    std::string calibration_file_name;


    // Add policy controller for Active Inference
    std::unique_ptr<PolicyController> policy_controller;
    
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
        Bind(peak_width_tolerance, "PeakWidthTolerance"); // Width of the peak for sharp peak detection in percentage of deviance history
        Bind(time_window, "TimeWindow");
        Bind(led_intensity, "LedIntensity");
        Bind(led_color_eyes, "LedColorEyes");
        Bind(led_color_mouth, "LedColorMouth");
    
        Bind(torque, "Torque");
        Bind(control_mode, "ControlMode"); 
        Bind(deviation_history_length, "DeviationHistoryLength");
        
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
        
        // Calibration parameters
        Bind(calibration_mode, "CalibrationMode");
 


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
      

        number_deviations_per_time_window.set_name("NumberDeviationsPerTimeWindow");
        number_deviations_per_time_window.copy(deviation); // Ensure size matches present_position
        number_deviations_per_time_window.reset(); // Initialize to zero

        time_window_start = GetTime();
        allowed_deviance.copy(allowed_deviance_input); // Initialize allowed_deviance with input

        // Initialize circular buffer for deviation history
        int max_history_length = deviation_history_length.as_int();
        deviation_history.set_name("DeviationHistory");
        deviation_history = matrix(max_history_length, deviation.size()); // Pre-allocate full buffer
        deviation_history.reset(); // Initialize to zero
        history_head = 0;
        history_count = 0;
        deviation_history.set_name("DeviationHistory");
        // Note: deviation_history will be properly initialized in first tick when deviation size is known
        
        // Initialize policy controller for Active Inference
        policy_controller = std::make_unique<PolicyController>(this, force_output, goal_position_out, torque,
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
        
        // Get the path of current script
        std::string script_path = __FILE__;
        // Remove filename to get directory
        size_t last_slash = script_path.find_last_of("/\\");
        std::string script_dir = (last_slash != std::string::npos) ? script_path.substr(0, last_slash) : ".";
        calibration_file_name = script_dir + "/VFE_calibration_data.json";

    
        
        if (calibration_mode) {
            is_calibrating = true;
            Debug("ForceCheck: Calibration mode enabled - learning parameters will be saved to " + calibration_file_name);

        } else {
            is_calibrating = false;
            Debug("ForceCheck: Normal operation mode - loading calibrated parameters from " + calibration_file_name);
        }
    }
    
    const char* GetModeNameByIndex(int index) {
        if (index == 0) return "Normal";
        if (index == 1) return "Avoidant";
        if (index == 2) return "Compliant";
        return "Unknown";
    }
    
    const char* GetPolicyNameByIndex(int index) {
        if (index == 0) return "CONTINUE";
        if (index == 1) return "STOP";
        if (index == 2) return "REFLECT";
        if (index == 3) return "COMPLY";
        return "Unknown";
    }

    // Circular buffer helper methods
    double GetRecentDeviation(int motor_idx, int steps_back) {
        if (steps_back >= history_count || motor_idx >= deviation_history.cols()) return 0.0;
        
        int max_history_length = deviation_history_length.as_int();
        int read_idx = (history_head - 1 - steps_back + max_history_length) % max_history_length;
        return deviation_history(read_idx, motor_idx);
    }
    
    int GetHistoryCount() {
        return history_count;
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
            
            // Feature 5: Position-dependent load factor (gravitational model)
            // For typical servo configuration: minimal load at 180°, peak load at 90°
            double angle_rad = present_position(i) * M_PI / 180.0; // Convert to radians
            double gravitational_load = abs(sin(angle_rad - M_PI)); // Minimal load at 180°, peak at 90°
            
            // Alternative: distance from minimal load position
            // double load_distance = abs(present_position(i) - 180.0) / 90.0; // Distance from 180°
            
            movement_context_features(base_idx + 4) = gravitational_load;
            
            // Feature 6: Movement effort (distance * load)
            double movement_effort = dist_to_goal * gravitational_load;
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
            precision_weights(i) = (1.0 - vfe_learning_rate) * precision_weights(i) + vfe_learning_rate * target_precision;
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
    
    // Calibration methods
    void SaveCalibrationData() {
        if (calibration_sample_count == 0) {
            Warning("ForceCheck: No calibration samples collected, cannot save calibration data.");
            return;
        }
        
        // Create JSON structure
        std::string json_content = "{\n";
        json_content += "  \"calibration_info\": {\n";
        json_content += "    \"sample_count\": " + std::to_string(calibration_sample_count) + ",\n";
        json_content += "    \"calibration_duration\": " + std::to_string(GetTime() - calibration_start_time) + ",\n";
        json_content += "    \"motor_count\": " + std::to_string(deviation.size()) + ",\n";
        json_content += "    \"vfe_enabled\": " + std::string(use_uncertainty_estimation ? "true" : "false") + "\n";
        json_content += "  },\n";
        
        if (use_uncertainty_estimation && calibration_precision_accumulator.size() > 0) {
            // Calculate average values from accumulated data
            matrix avg_precision_weights = calibration_precision_accumulator;
            matrix avg_adaptive_thresholds = calibration_threshold_accumulator;
            
            for (int i = 0; i < avg_precision_weights.size(); i++) {
                avg_precision_weights(i) /= calibration_sample_count;
            }
            for (int i = 0; i < avg_adaptive_thresholds.size(); i++) {
                avg_adaptive_thresholds(i) /= calibration_sample_count;
            }
            
            // Save precision weights
            json_content += "  \"precision_weights\": [";
            for (int i = 0; i < avg_precision_weights.size(); i++) {
                if (i > 0) json_content += ", ";
                json_content += std::to_string(avg_precision_weights(i));
            }
            json_content += "],\n";
            
            // Save adaptive thresholds
            json_content += "  \"adaptive_thresholds\": [";
            for (int i = 0; i < avg_adaptive_thresholds.size(); i++) {
                if (i > 0) json_content += ", ";
                json_content += std::to_string(avg_adaptive_thresholds(i));
            }
            json_content += "],\n";
        } else {
            // Basic calibration without VFE data
            json_content += "  \"precision_weights\": [],\n";
            json_content += "  \"adaptive_thresholds\": [],\n";
        }
        
        // Save VFE learning parameters
        json_content += "  \"vfe_parameters\": {\n";
        json_content += "    \"vfe_learning_rate\": " + std::to_string((double)vfe_learning_rate) + ",\n";
        json_content += "    \"uncertainty_threshold_scale\": " + std::to_string((double)uncertainty_threshold_scale) + ",\n";
        json_content += "    \"precision_update_rate\": " + std::to_string((double)precision_update_rate) + "\n";
        json_content += "  }\n";
        json_content += "}\n";
        
        // Write to file
        std::ofstream file(calibration_file_name);
        if (file.is_open()) {
            file << json_content;
            file.close();
            Debug("ForceCheck: Calibration data saved to " + calibration_file_name);
            Debug("ForceCheck: Saved " + std::to_string(calibration_sample_count) + " samples over " + 
                  std::to_string(GetTime() - calibration_start_time) + " seconds");
        } else {
            Error("ForceCheck: Failed to open calibration file for writing: " + calibration_file_name);
        }
    }
    
    void LoadCalibrationData() {
        std::ifstream file(calibration_file_name);
        if (!file.is_open()) {
            Warning("ForceCheck: Calibration file not found: " + calibration_file_name + ", using default parameters.");
            return;
        }
        
        std::string line, content;
        while (std::getline(file, line)) {
            content += line + "\n";
        }
        file.close();
        
        // Simple JSON parsing for precision weights
        size_t precision_start = content.find("\"precision_weights\": [");
        if (precision_start != std::string::npos) {
            precision_start = content.find("[", precision_start) + 1;
            size_t precision_end = content.find("]", precision_start);
            
            if (precision_end != std::string::npos) {
                std::string precision_str = content.substr(precision_start, precision_end - precision_start);
                
                // Parse comma-separated values
                std::stringstream ss(precision_str);
                std::string value;
                int idx = 0;
                
                while (std::getline(ss, value, ',') && idx < precision_weights.size()) {
                    // Remove whitespace
                    value.erase(0, value.find_first_not_of(" \t\n\r"));
                    value.erase(value.find_last_not_of(" \t\n\r") + 1);
                    
                    try {
                        precision_weights(idx) = std::stod(value);
                        idx++;
                    } catch (const std::exception& e) {
                        Warning("ForceCheck: Error parsing precision weight " + std::to_string(idx) + ": " + value);
                    }
                }
                Debug("ForceCheck: Loaded " + std::to_string(idx) + " precision weights from calibration file");
            }
        }
        
        // Simple JSON parsing for adaptive thresholds
        size_t threshold_start = content.find("\"adaptive_thresholds\": [");
        if (threshold_start != std::string::npos) {
            threshold_start = content.find("[", threshold_start) + 1;
            size_t threshold_end = content.find("]", threshold_start);
            
            if (threshold_end != std::string::npos) {
                std::string threshold_str = content.substr(threshold_start, threshold_end - threshold_start);
                
                // Parse comma-separated values
                std::stringstream ss(threshold_str);
                std::string value;
                int idx = 0;
                
                while (std::getline(ss, value, ',') && idx < adaptive_threshold_base.size()) {
                    // Remove whitespace
                    value.erase(0, value.find_first_not_of(" \t\n\r"));
                    value.erase(value.find_last_not_of(" \t\n\r") + 1);
                    
                    try {
                        adaptive_threshold_base(idx) = std::stod(value);
                        idx++;
                    } catch (const std::exception& e) {
                        Warning("ForceCheck: Error parsing adaptive threshold " + std::to_string(idx) + ": " + value);
                    }
                }
                Debug("ForceCheck: Loaded " + std::to_string(idx) + " adaptive thresholds from calibration file");
            }
        }
        
        Debug("ForceCheck: Calibration data loaded from " + calibration_file_name);
    }



    void Tick()
    {
        deviation.copy(current_prediction);
        deviation.subtract(present_current);



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
                    expected_free_energy_output = matrix(4); // 4 policies: CONTINUE, STOP, REFLECT, COMPLY
                    
                    epistemic_uncertainty.set(0.0);
                    aleatoric_uncertainty.set(0.0);
                    precision_weights.set(1.0);
                    expected_free_energy_output.set(0.0);
                    
                    last_precision_update_time = GetTime();
                    
                    Debug("VFE: Initialized matrices for " + std::to_string(deviation.size()) + " motors");
                }
                
                // Initialize calibration accumulators if in calibration mode
                if (is_calibrating) {
                    calibration_precision_accumulator = matrix(deviation.size());
                    calibration_threshold_accumulator = matrix(deviation.size());
                    calibration_precision_accumulator.reset();
                    calibration_threshold_accumulator.reset();
                    calibration_start_time = GetTime();
                    Debug("ForceCheck: Calibration accumulators initialized");
                } else {
                    // Load previously calibrated parameters
                    LoadCalibrationData();
                }
            }
            
            firstTick = false; // Set to false after first initialization
        }
        
        // Add to circular buffer deviation history
        if (deviation.size() > 0) {
            // Only reinitialize if buffer size parameters have changed
            int max_history_length = deviation_history_length.as_int();
            if (deviation_history.rows() != max_history_length || deviation_history.cols() != deviation.size()) {
                Debug("ForceCheck: Reinitializing deviation history buffer - rows: " + 
                      std::to_string(deviation_history.rows()) + " -> " + std::to_string(max_history_length) +
                      ", cols: " + std::to_string(deviation_history.cols()) + " -> " + std::to_string(deviation.size()));
                
                deviation_history = matrix(max_history_length, deviation.size());
                deviation_history.reset();
                history_head = 0;
                history_count = 0;
            }
            
            // Write to current head position
            for (int i = 0; i < deviation.size(); i++) {
                deviation_history(history_head, i) = deviation(i);
            }
            deviation_history.print();
            
            // Advance head (circular)
            history_head = (history_head + 1) % max_history_length;
            
            // Update count (saturate at max)
            if (history_count < max_history_length) {
                history_count++;
            }
        }
        
        
        // Handle manual policy switching via parameter (for testing/debugging)
        static int last_manual_mode_setting = -1;
        if (control_mode.as_int() != last_manual_mode_setting) {
            if (!automatic_mode_switching_enabled || last_manual_mode_setting == -1) { // Allow manual override or initial set
                Debug("ForceCheck: Manual policy override to policy " + std::to_string(control_mode.as_int()) + 
                      " (" + policy_controller->GetPolicyName(control_mode.as_int()) + ")");
                // Note: Manual override will be applied in the policy execution section below
            }
            last_manual_mode_setting = control_mode.as_int();
        }
        
        // Force CONTINUE policy during calibration (policy 0)
        if (is_calibrating && last_manual_mode_setting != 0) {
            Debug("ForceCheck: Calibration mode active - forcing CONTINUE policy");
            control_mode = 0;
            last_manual_mode_setting = 0;
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

            // Collect calibration data if in calibration mode (independent of VFE)
            if (is_calibrating && use_uncertainty_estimation) {
                // Accumulate precision weights and adaptive thresholds for averaging
                for (int i = 0; i < precision_weights.size(); i++) {
                    calibration_precision_accumulator(i) += precision_weights(i);
                }
                for (int i = 0; i < adaptive_threshold_base.size(); i++) {
                    calibration_threshold_accumulator(i) += adaptive_threshold_base(i);
                }
                calibration_sample_count++;
                
                // Periodically save calibration data and report progress
                if (calibration_sample_count % 100 == 0) {
                    Debug("ForceCheck: Calibration progress - " + std::to_string(calibration_sample_count) + 
                          " samples collected over " + std::to_string(GetTime() - calibration_start_time) + " seconds");
                    
                    // Save intermediate calibration data every 500 samples
                    if (calibration_sample_count % 500 == 0) {
                        SaveCalibrationData();
                        Debug("ForceCheck: Intermediate calibration data saved");
                    }
                }
            } else if (is_calibrating) {
                // Basic calibration mode without VFE - just count samples for timing
                calibration_sample_count++;
                if (calibration_sample_count % 100 == 0) {
                    Debug("ForceCheck: Basic calibration progress - " + std::to_string(calibration_sample_count) + 
                          " samples collected over " + std::to_string(GetTime() - calibration_start_time) + " seconds");
                }
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
            
            // Use VFE-enhanced thresholds for active inference controller BEFORE computing EFE
            active_inference_controller->SetAllowedDeviance(
                use_uncertainty_estimation ? adaptive_threshold_base : allowed_deviance_input);
            active_inference_controller->UpdatePrecisionWeights(deviation, motor_in_motion,
                                                                number_deviations_per_time_window, deviation_history,
                                                                history_count, history_head);
            
            // Update temporal patterns BEFORE calculating EFE
            active_inference_controller->UpdateTemporalPatterns(deviation, deviation_history, torque,
                                                                history_count, history_head);
            
            active_inference_controller->CalculateExpectedFreeEnergy(deviation, present_position,
                                                                     goal_position_in);

            // Update the output matrix with expected free energy values
            if (use_uncertainty_estimation && expected_free_energy_output.size() == 4) {
                matrix& efe = active_inference_controller->GetExpectedFreeEnergy();
                if (efe.size() >= 4) {
                    expected_free_energy_output.copy(efe);
                }
            }

            // Select and execute policy exactly once per tick after EFE is computed
            int selected_policy = active_inference_controller->SelectMode();
            int current_policy = policy_controller->GetCurrentPolicy();

            if (automatic_mode_switching_enabled && !refractory_period && !is_calibrating) {
                if (selected_policy != current_policy) {
                    matrix& efe = active_inference_controller->GetExpectedFreeEnergy();
                    Debug("ForceCheck: Active inference switching from '" +
                          std::string(GetPolicyNameByIndex(current_policy)) +
                          "' to '" + std::string(GetPolicyNameByIndex(selected_policy)) +
                          "' (EFE: " + efe.json() + ")");

                    policy_controller->SetCurrentPolicy(selected_policy);
                    refractory_period = true;
                    refractory_start_time = GetTime();
                } else {
                    matrix& efe = active_inference_controller->GetExpectedFreeEnergy();
                    Debug("ForceCheck: Policy selection - Current: " + std::string(GetPolicyNameByIndex(current_policy)) +
                          " (EFE: " + efe.json() + ")");
                }
            }

            // Execute the selected (or current) policy after selection
            policy_controller->ExecutePolicy(automatic_mode_switching_enabled ? policy_controller->GetCurrentPolicy() : selected_policy,
                                             deviation,
                                             present_position,
                                             goal_position_in,
                                             pullback_amount,
                                             GetTime());
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

        
    }
};

INSTALL_CLASS(ForceCheckActiveInference);

