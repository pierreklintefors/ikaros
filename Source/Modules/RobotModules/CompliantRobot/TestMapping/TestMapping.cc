#include "ikaros.h"
#include <random> // Include the header file that defines the 'Random' function
#include <fstream>
#include <vector>
#include <memory>
#include "json.hpp"
#include <string>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <thread>
#include <sys/wait.h>
#include <atomic>
#include <limits> // Add this for numeric_limits
#include <sstream> // Add this for stringstream
#include <iomanip> // Add this for std::fixed, std::setprecision
#include <filesystem> // Add this for directory creation

// Include nanoflann
#include "nanoflann.hpp" // Adjust path if needed, e.g., "libs/nanoflann.hpp"

using namespace ikaros;
using json = nlohmann::json; // Alias for convenience

// --- Nanoflann Point Cloud Adaptor ---
// Structure to hold the NN feature data and target currents
struct PointCloud
{
	// Feature points (inner vector size determined dynamically)
	std::vector<std::vector<float>>  pts; // Standardized feature data
	// Corresponding target current values for each point
	std::vector<float> target_currents;
    // Feature names loaded from CSV header (excluding target)
    std::vector<std::string> feature_names;
    // Number of features (dimensionality)
    size_t num_features = 0;

    // Normalization data (means and stds for features only)
    std::vector<float> feature_means;
    std::vector<float> feature_stds;
    bool normalization_enabled = true; // Flag if means/stds were loaded
    // <<< EDIT 8: Add members for target normalization >>>
    float target_mean = 0.0f;
    float target_std = 1.0f; // Default to 1 to avoid division by zero if not loaded
    // <<< END EDIT 8 >>>


	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	// Since pts[idx] is std::vector<float>, we can just return pts[idx][dim]
	// Dimensionality is checked externally based on num_features
	inline float kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		// Basic bounds check for safety, though dim should be < num_features
		if (idx < pts.size() && dim < pts[idx].size()) {
			return pts[idx][dim];
		}
		// Return a default or handle error if index/dim is out of bounds
		// For simplicity, returning 0. Proper error handling might be needed.
		// Consider throwing or logging an error here in production.
		std::cerr << "Out-of-bounds access in kdtree_get_pt: idx=" << idx << ", dim=" << dim << ", point count=" << pts.size() << std::endl;
		return 0.0f;
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	// Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	// Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};
// --- End Nanoflann ---

#define SHM_NAME "/ikaros_ann_shm"
#define FLOAT_COUNT 21  // 19 inputs + 2 predictions
#define MEM_SIZE ((FLOAT_COUNT * sizeof(float)) + sizeof(bool)*2)  // Add explicit size calculation including two flags
#define NN_DIMS 17  // 9 base features + 4 features per servo * 2 servos (for Torso)

// Structure for shared memory between Python and C++
struct SharedData
{
    float data[FLOAT_COUNT]; 
    bool new_data;           // Flag for synchronization
    bool shutdown_flag;
    // Add padding to ensure proper alignment
    char padding[sizeof(float) - (sizeof(bool)*2 % sizeof(float))];
};

class TestMapping: public Module
{
 

    // Inputs
    matrix present_position;
    matrix present_current;
    matrix gyro;
    matrix accel;
    matrix eulerAngles;

    // Outputs
    matrix goal_current;
    matrix goal_position_out;
    

    // Internal
    matrix current_controlled_servos;
    matrix max_present_current;
    matrix min_moving_current;
    matrix min_torque_current;
    matrix overshot_goal;
    matrix overshot_goal_temp;
    matrix position_transitions;
    std::vector<std::shared_ptr<std::vector<float>>> moving_trajectory;
    matrix approximating_goal;
    matrix reached_goal;
    dictionary current_coefficients;
    matrix coeffcient_matrix;
    
   
    // Parameters
    parameter num_transitions;
    parameter min_limits;
    parameter max_limits;
    parameter robotType;
    parameter prediction_model;
    parameter current_increment_parameter;    
    parameter log_deviance_data;
    // Internal
    std::random_device rd;
    
    int number_transitions; //Will be used to add starting and ending position
    int position_margin = 2;
    int transition = 0;
    int starting_current = 30;
    int current_limit = 2000;
    bool find_minimum_torque_current = false;
    int current_increment;
  
    int unique_id;
    bool second_tick = false;
    bool first_start_position = true;
    std::string time_stamp_str_no_dots;
    int num_input_data = 0;
    matrix transition_start_time;
    matrix transition_duration;
    double time_prev_position;
    matrix model_prediction;
    matrix model_prediction_start;

    std::vector<std::string> servo_names;

    matrix initial_currents;
    matrix current_differences;

    // Shared memory variables
    int shm_fd;
    SharedData* shared_data;

    std::string shm_name;
    
 
    const double TIMEOUT_DURATION = 4.0;    // Timeout before incrementing current
        // Amount to increment current during timeout
    
  
    matrix timeout_occurred;

    
    matrix predicted_goal_current;
    matrix prediction_error;

    matrix starting_positions; // New matrix to store starting positions

    // Add a member variable to hold the child process ID
    pid_t child_pid = -1;

    // --- Nearest Neighbor data ---
    // Point clouds holding feature data and target currents
    PointCloud nn_tilt_point_cloud;
    PointCloud nn_pan_point_cloud;

    // Define the k-d tree index type using the PointCloud adaptor
    // <<< EDIT 7: Use dynamic dimensionality (-1) for the k-d tree adaptor >>>
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud>, // Distance metric (L2 = Euclidean)
        PointCloud,                                    // Dataset adaptor
        -1                                             // Dimensionality (-1 for dynamic)
        >;
    // <<< END EDIT 7 >>>

    // Unique pointers to hold the built k-d tree indices
    std::unique_ptr<my_kd_tree_t> nn_tilt_index;
    std::unique_ptr<my_kd_tree_t> nn_pan_index;

 

    // <<< EDIT 1: Add members for deviance logging >>>
    std::ofstream deviance_log_file;       // File stream for logging
    std::stringstream deviance_log_stream; // Stringstream buffer for JSON content
     // Flag to enable logging logic
    int deviance_log_tick_counter = 0;     // Counter for periodic flushing
    const int DEVIANCE_LOG_FLUSH_INTERVAL = 2; // How often to flush (ticks)
    bool is_first_deviance_log_entry = true; // To handle commas in JSON array
    // <<< END EDIT 1 >>>

    matrix RandomisePositions(int num_transitions, matrix min_limits, matrix max_limits, std::string robotType)
    {
        // Initialize the random number generator with the seed
        std::mt19937 gen(rd());
        int num_columns = (robotType == "Torso") ? 2 : 12;
        matrix generated_positions(num_transitions, present_position.size());
        generated_positions.set(180); // Neutral position
        // set pupil servos to 12 of all rows
        for (int i = 0; i < generated_positions.rows(); i++)
        {
            generated_positions(i, 4) = 12;
            generated_positions(i, 5) = 12;
        }
        if (num_columns != min_limits.size() || num_columns != max_limits.size())
        {
            Error("Min and Max limits must have the same number of columns as the current controlled servos in the robot type (2 for Torso, 12 for full body)");
            return -1;
        }

        for (int i = 0; i < num_transitions; i++)
        {
            for (int j = 0; j < num_columns; j++)
            {
                std::uniform_real_distribution<double> distr(min_limits(j), max_limits(j));
                generated_positions(i, j) = int(distr(gen));
                if (i == num_transitions - 1)
                {
                    generated_positions(i, j) = 180;
                }
            }
        }

        return generated_positions;
    }

    void ReachedGoal(matrix present_position, matrix goal_positions, matrix reached_goal, int margin){
        // Add safety check
        if (present_position.size() != goal_positions.size() || 
            present_position.size() <= 0 || 
            reached_goal.size() <= 0) {
            Error("Invalid matrix dimensions in ReachedGoal");
            return;
        }
        
        for (int i = 0; i < current_controlled_servos.size(); i++){
            int servo_idx = current_controlled_servos(i);
            // Add bounds check
            if (servo_idx < 0 || servo_idx >= present_position.size() || 
                servo_idx >= goal_positions.size() || 
                servo_idx >= reached_goal.size()) {
                Error("Servo index out of bounds: " + std::to_string(servo_idx));
                continue;
            }
            
            if (reached_goal(current_controlled_servos(i)) == 0 &&
                abs(present_position(current_controlled_servos(i)) - goal_positions(current_controlled_servos(i))) < margin){

                reached_goal(current_controlled_servos(i)) = 1;

                double current_time_ms = GetTime();
                
                transition_duration(transition, i) = current_time_ms - transition_start_time(transition, i);
                // Reset start time for future transitions
     

            }
          

        }
    }

    matrix ApproximatingGoal(matrix &present_position,  matrix goal_position, int margin){
        matrix previous_position = present_position.last();
        for (int i =0; i < current_controlled_servos.size(); i++){
            //Checking if distance to goal is decreasing
            if (abs(goal_position(current_controlled_servos(i)) - present_position(current_controlled_servos(i))) < 0.97*abs(goal_position(current_controlled_servos(i))-previous_position(current_controlled_servos(i)))){
                Debug( "Approximating Goal");
                approximating_goal(current_controlled_servos(i)) = 1;
                
            }
            else{
                approximating_goal(current_controlled_servos(i)) = 0;
            }
        }
        return approximating_goal;
    }   
    
    
    
    int GenerateRandomNumber(int min, int max){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distr(min, max);
        return distr(gen);
    }

    // FUnction to print a progress bar of the current transition
    void PrintProgressBar(int transition, int number_transitions){
        int barWidth = 70;
        float progress = (float)transition/number_transitions;
        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r" << std::endl;
        std::cout.flush();
    }

    // Helper function to find column index case-insensitive
    int findColumnCaseInsensitive( ikaros::matrix &m,  std::string &label)
    {
        std::string lowerLabel = label;
        std::transform(lowerLabel.begin(), lowerLabel.end(), lowerLabel.begin(), ::tolower);

        auto labels = m.labels(1); // Get column labels
        for (size_t i = 0; i < labels.size(); i++)
        {
            std::string currentLabel = labels[i];
            std::transform(currentLabel.begin(), currentLabel.end(), currentLabel.begin(), ::tolower);
            if (currentLabel == lowerLabel)
            {
                return i;
            }
        }
        throw std::runtime_error("Label not found: " + label);
    }

    std::vector<double> extractModelParameters(ikaros::matrix &m, std::string &model_name)
    {
        // Define all possible parameters
        std::vector<std::string> parameter_labels = {
            "CurrentMean",
            "CurrentStd",
            "betas_linear[DistanceToGoal]",
            "betas_linear[Position]",
            "betas_quad[DistanceToGoal_squared]",
            "betas_quad[Position_squared]",
            "intercept"};

        std::vector<double> values(parameter_labels.size(), 0.0);

        try
        {
            for (size_t i = 0; i < parameter_labels.size(); i++)
            {
                // Skip quadratic terms if model is linear
                if (model_name != "Quadratic" &&
                    (parameter_labels[i].find("quad") != std::string::npos ||
                     parameter_labels[i].find("squared") != std::string::npos))
                {
                    values[i] = 0.0;
                    continue;
                }

                try
                {
                    values[i] = static_cast<double>(m(0, findColumnCaseInsensitive(m, parameter_labels[i])));
                }
                catch (const std::runtime_error &e)
                {
                    // Set to 0.0 for non-required parameters in linear model
                    values[i] = 0.0;
                }
            }
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Error extracting parameters: " + std::string(e.what()));
        }

        return values;
    }
    matrix CreateCoefficientsMatrix(dictionary coefficients, matrix current_controlled_servos, std::string model, std::vector<std::string> servo_names){
        // mapping that defines number of coefficients for each model
        //  Mapping that defines number of coefficients for each model
        // std::map<std::string, int> model_coefficients = {{"Linear", 5}, {"Quadratic", 7}}; // Removed hardcoded map
        // int num_coefficients = model_coefficients[model]; // Removed hardcoded size determination

        // Determine the number of coefficients dynamically from the first servo
        if (current_controlled_servos.size() == 0) {
            Error("current_controlled_servos is empty. Cannot determine coefficient matrix size.");
            return matrix(); // Return empty matrix
        }

        std::string first_servo_name_base = servo_names[current_controlled_servos(0)];
        //remove Neck from the servo name
        if (first_servo_name_base.find("Neck") != std::string::npos)
        {
            first_servo_name_base = first_servo_name_base.substr(4);
        }

        dictionary first_servo_coeffs;
        try {
            first_servo_coeffs = coefficients[model][first_servo_name_base];
        } catch (const std::exception& e) {
             Error("Could not find coefficients for model '" + model + "' and base servo '" + first_servo_name_base + "': " + e.what());
             return matrix(); // Return empty matrix
        }


        int num_coefficients = 0;
        for (auto& coeff : first_servo_coeffs) {
            if (coeff.first != "sigma") {
                num_coefficients++;
            }
        }

        if (num_coefficients == 0) {
            Error("No non-sigma coefficients found for model '" + model + "' and servo '" + first_servo_name_base + "'. Cannot create matrix.");
            return matrix(); // Return empty matrix
        }


        // Initialize matrix with coefficients for each servo using dynamic size
        matrix coefficients_matrix(current_controlled_servos.size(), num_coefficients);
        
        // Iterate through each servo
        for (int i = 0; i < current_controlled_servos.size(); i++)
        {
            std::string servo_name = servo_names[current_controlled_servos(i)];
            //remove Neck from the servo name
            if (servo_name.find("Neck") != std::string::npos)
            {
                servo_name = servo_name.substr(4);
            }
            

            // Get the coefficients dictionary for this servo
            dictionary servo_coeffs;
             try {
                servo_coeffs = coefficients[model][servo_name];
             } catch (const std::exception& e) {
                 Warning("Could not find coefficients for model '" + model + "' and servo '" + servo_name + "': " + e.what() + ". Skipping this servo.");
                 continue; // Skip to the next servo if coefficients are missing
             }


            // Skip "sigma" if it exists and iterate over the coefficient values
            int coeff_idx = 0;
            for ( auto &coeff : servo_coeffs)
            {
                if (coeff.first != "sigma")
                { // Skip sigma parameter
                    // Ensure we don't write out of bounds if a servo has fewer coefficients than the first one
                    if (coeff_idx < num_coefficients) {
                        coefficients_matrix(i, coeff_idx) = (coeff.second).as_float();
                        if (i == 0) // Add labels only based on the first servo's coefficients
                            coefficients_matrix.push_label(1, coeff.first);
                    } else {
                         Warning("Servo '" + servo_name + "' has more coefficients than expected based on first servo '" + first_servo_name_base + "'. Ignoring extra coefficient: " + coeff.first);
                    }
                    coeff_idx++;
                }
            }
             // Check if this servo had fewer coefficients than expected
             if (coeff_idx < num_coefficients) {
                 Warning("Servo '" + servo_name + "' has fewer coefficients (" + std::to_string(coeff_idx) + ") than expected (" + std::to_string(num_coefficients) + "). Remaining coefficients in row " + std::to_string(i) + " will be uninitialized (likely 0).");
             }
        }
        coefficients_matrix.set_name("CoefficientMatrix");
       

        return coefficients_matrix;
    }

    matrix SetGoalCurrent(matrix present_current, int increment, int limit, 
                         matrix position, matrix goal_position, int margin, 
                         matrix coefficient_matrix, const dictionary& current_coefficients, // Added dictionary param
                         std::string model_name, matrix model_prediction)
    {
        matrix current_output(present_current.size());
        current_output.set(0);

        // Early validation
        if (present_current.size() != current_output.size() ||
            position.size() != present_current.size() ||
            goal_position.size() != present_current.size() ||
            model_prediction.size() != present_current.size() ) // Added more checks
        {
            Error("Input/output matrix size mismatch in SetGoalCurrent");
            return current_output;
        }
        // Also check sensor data if needed for NN
        if (model_name == "NearestNeighbor" && (!gyro.connected() || !accel.connected() || !eulerAngles.connected())) {
             Error("Gyro, Accel, and EulerAngles must be connected for NearestNeighbor model.");
            return current_output; // Or handle differently
        }

        // Map to store column indices for coefficient names
        std::map<std::string, int> coeff_indices;
        bool indices_found = false;

        // Find column indices by label if we're using Linear or Quadratic models
        if ((model_name == "Linear" || model_name == "Quadratic") && coefficient_matrix.size_y() > 0) {
            // Get column labels
            auto labels = coefficient_matrix.labels(1);
            if (labels.size() > 0) {
                // Populate the map with label->index pairs
                for (size_t i = 0; i < labels.size(); i++) {
                    coeff_indices[labels[i]] = i;
                }
                indices_found = true;
                
                // Debug output to verify labels were found
                std::string found_labels = "Found coefficient labels: ";
                for (const auto& pair : coeff_indices) {
                    found_labels += pair.first + "=" + std::to_string(pair.second) + " ";
                }
                Debug(found_labels);
            } else {
                Error("No column labels found in coefficient matrix. Cannot identify coefficients.");
            }
        }

        // Only process servos that are current-controlled
        for (int i = 0; i < current_controlled_servos.size(); i++)
        {
            int servo_idx = current_controlled_servos(i);

            // Skip processing if servo has reached its goal
            if (abs(goal_position(servo_idx) - position(servo_idx)) <= margin)
            {
                // Maybe set current to 0 or a holding value? For now, skip.
                continue;
            }

            // Calculate estimated current based on model type
            double estimated_current = 0.0;

            if (model_name == "ANN")
            {
                // Use existing ANN output for this servo
                // NOTE: ANN output is updated elsewhere (in Tick) and stored in model_prediction/model_prediction_start
                // This function *might* not be the right place to set ANN current directly,
                // but we use the value already stored in model_prediction by the Tick logic.
                 estimated_current = model_prediction[servo_idx]; // Assuming Tick already populated this
            }
            else if (model_name == "Linear" || model_name == "Quadratic")
            {
                // Check if indices were successfully found earlier
                if (!indices_found) {
                    // Fallback to order-based access if no labels were found
                    Warning("No coefficient labels found, using fallback order-based access method");
                    
                    // Ensure coefficient matrix row is valid
                    if (coefficient_matrix.rows() <= i || coefficient_matrix.cols() < 7) {
                        Error("Coefficient matrix invalid for servo index " + std::to_string(i));
                        continue; // Skip this servo
                    }
                    
                    // Use the manually provided fixed positions (from original function)
                    double current_mean = coefficient_matrix(i, 0); // Assuming column 0 is CurrentMean
                    double current_std = coefficient_matrix(i, 1);  // Assuming column 1 is CurrentStd
                    double position_mean = coefficient_matrix(i, 2); // Assuming column 2 is PositionMean
                    double position_std = coefficient_matrix(i, 3); // Assuming column 3 is PositionStd
                    double distance_mean = coefficient_matrix(i, 4); // Assuming column 4 is DistanceMean
                    double distance_std = coefficient_matrix(i, 5); // Assuming column 5 is DistanceStd
                    
                    // Check for zero standard deviation
                    if (current_std == 0) {
                        Warning("Current standard deviation is zero for servo index " + std::to_string(i) + ". Using mean as estimate.");
                        estimated_current = current_mean;
                    } else {
                        // Calculate standardized inputs
                        double distance_to_goal = goal_position(servo_idx) - position(servo_idx);
                        
                        // Prevent division by zero
                        double std_position = (position_std != 0) ? (position(servo_idx) - position_mean) / position_std : 0.0;
                        double std_distance = (distance_std != 0) ? (distance_to_goal - distance_mean) / distance_std : 0.0;
                        
                        // Calculate linear terms
                        estimated_current = coefficient_matrix(i, 6) + // intercept (index 6)
                                            coefficient_matrix(i, 2) * std_distance + // betas_linear[DistanceToGoal] (index 2)
                                            coefficient_matrix(i, 3) * std_position;  // betas_linear[Position] (index 3)
                        
                        // Add quadratic terms if applicable
                        if (model_name == "Quadratic") {
                            estimated_current += coefficient_matrix(i, 4) * std::pow(std_distance, 2) + // quad distance (index 4)
                                                coefficient_matrix(i, 5) * std::pow(std_position, 2);  // quad position (index 5)
                        }
                        
                        // Unstandardize the result
                        estimated_current = estimated_current * current_std + current_mean;
                    }
                } else {
                    // Use the dynamically found indices from coefficient labels
                    // Check that all required indices exist
                    if (!coeff_indices.count("CurrentMean") || !coeff_indices.count("CurrentStd") ||
                        !coeff_indices.count("PositionMean") || !coeff_indices.count("PositionStd") ||
                        !coeff_indices.count("DistanceMean") || !coeff_indices.count("DistanceStd") ||
                        !coeff_indices.count("intercept") || !coeff_indices.count("betas_linear[DistanceToGoal]") ||
                        !coeff_indices.count("betas_linear[Position]")) {
                        
                        Error("Missing required coefficient labels for Linear model. Check coefficient matrix.");
                        continue; // Skip this servo
                    }
                    
                    // Additional check for Quadratic model
                    if (model_name == "Quadratic" && 
                        (!coeff_indices.count("betas_quad[DistanceToGoal_squared]") || 
                         !coeff_indices.count("betas_quad[Position_squared]"))) {
                        
                        Error("Missing required quadratic coefficient labels. Check coefficient matrix.");
                        continue; // Skip this servo
                    }
                    
                    // Access coefficients using the map
                    double current_mean = coefficient_matrix(i, coeff_indices["CurrentMean"]);
                    double current_std = coefficient_matrix(i, coeff_indices["CurrentStd"]);
                    double position_mean = coefficient_matrix(i, coeff_indices["PositionMean"]);
                    double position_std = coefficient_matrix(i, coeff_indices["PositionStd"]);
                    double distance_mean = coefficient_matrix(i, coeff_indices["DistanceMean"]);
                    double distance_std = coefficient_matrix(i, coeff_indices["DistanceStd"]);

                    // Check for zero standard deviation for current (critical)
                    if (current_std == 0) {
                        Warning("Current standard deviation is zero for servo index " + std::to_string(i) + ". Using mean as estimate.");
                        estimated_current = current_mean;
                    } else {
                        // Calculate standardized inputs
                        double distance_to_goal = goal_position(servo_idx) - position(servo_idx);

                        // Handle potential division by zero if std devs are zero
                        double std_position = (position_std != 0) ? (position(servo_idx) - position_mean) / position_std : 0.0;
                        double std_distance = (distance_std != 0) ? (distance_to_goal - distance_mean) / distance_std : 0.0;

                        // Calculate linear terms using dynamically found indices
                        estimated_current = coefficient_matrix(i, coeff_indices["intercept"]) +
                                            coefficient_matrix(i, coeff_indices["betas_linear[DistanceToGoal]"]) * std_distance +
                                            coefficient_matrix(i, coeff_indices["betas_linear[Position]"]) * std_position;

                        // Add quadratic terms if applicable, using dynamically found indices
                        if (model_name == "Quadratic") {
                            estimated_current += coefficient_matrix(i, coeff_indices["betas_quad[DistanceToGoal_squared]"]) * std::pow(std_distance, 2) +
                                                coefficient_matrix(i, coeff_indices["betas_quad[Position_squared]"]) * std::pow(std_position, 2);
                        }

                        // Unstandardize the result
                        estimated_current = estimated_current * current_std + current_mean;
                    }
                }
            }
            // <<< EDIT 7: Add NearestNeighbor logic >>>
            else if (model_name == "NearestNeighbor") {
                 // Check if indices were built successfully in Init and normalization data is available
                 bool tilt_ready = nn_tilt_index && nn_tilt_point_cloud.normalization_enabled;
                 bool pan_ready = nn_pan_index && nn_pan_point_cloud.normalization_enabled;
                 bool index_ready = (servo_idx == 0 && tilt_ready) || (servo_idx == 1 && pan_ready);

                 if (!index_ready) {
                     Error("NN k-d tree index or normalization data not ready for servo " + std::to_string(servo_idx));
                     continue; // Skip NN estimation if index/data isn't ready
                 }

                 // Select the correct index and point cloud based on servo index
                 const std::unique_ptr<my_kd_tree_t>* current_index_ptr = nullptr;
                 const PointCloud* current_point_cloud_ptr = nullptr;

                 if (servo_idx == 0) { // NeckTilt
                    current_index_ptr = &nn_tilt_index;
                    current_point_cloud_ptr = &nn_tilt_point_cloud;
                 } else if (servo_idx == 1) { // NeckPan
                     current_index_ptr = &nn_pan_index;
                     current_point_cloud_ptr = &nn_pan_point_cloud;
                 } else {
                      Warning("NearestNeighbor model only implemented for NeckTilt (0) and NeckPan (1). Servo " + std::to_string(servo_idx) + " not handled.");
                      estimated_current = 0;
                      continue;
                 }

                 // Check if the selected point cloud has features
                 if (current_point_cloud_ptr->num_features == 0 || current_point_cloud_ptr->feature_names.empty()) {
                      Error("Selected PointCloud for servo " + std::to_string(servo_idx) + " has no features loaded.");
                      continue;
                 }
                 size_t num_features = current_point_cloud_ptr->num_features;

                 // <<< EDIT: Construct the *raw* query vector using only the 4 servo-specific features >>>
                 // The order MUST match the order defined and loaded in LoadCSVData:
                 // 0: Position, 1: DistToGoal, 2: GoalPosition, 3: StartPosition
                 size_t expected_num_features = 4; // We now expect exactly 4 features
                 if (current_point_cloud_ptr->num_features != expected_num_features) {
                      Error("PointCloud for servo " + std::to_string(servo_idx) +
                            " has unexpected number of features (" + std::to_string(current_point_cloud_ptr->num_features) +
                            "). Expected " + std::to_string(expected_num_features) + ". Check LoadCSVData filtering.");
                      continue;
                 }
                 std::vector<float> query_point_raw(expected_num_features);
                 bool query_build_ok = true;

                 // Feature 0: Position
                 if (present_position.size() > servo_idx) {
                     query_point_raw[0] = present_position(servo_idx);
                 } else {
                     Error("Present position index out of bounds for servo " + std::to_string(servo_idx));
                     query_build_ok = false;
                 }

                 // Feature 1: DistToGoal
                 if (query_build_ok && present_position.size() > servo_idx && goal_position_out.size() > servo_idx) {
                     query_point_raw[1] = goal_position_out(servo_idx) - present_position(servo_idx);
                 } else if (query_build_ok) {
                     Error("Present or goal position index out of bounds for DistToGoal for servo " + std::to_string(servo_idx));
                     query_build_ok = false;
                 }

                 // Feature 2: GoalPosition
                 if (query_build_ok && goal_position_out.size() > servo_idx) {
                     query_point_raw[2] = goal_position_out(servo_idx);
                 } else if (query_build_ok) {
                     Error("Goal position index out of bounds for servo " + std::to_string(servo_idx));
                     query_build_ok = false;
                 }

                 // Feature 3: StartPosition
                 if (query_build_ok && transition < starting_positions.rows() && starting_positions.cols() > i /* Use i (loop index) here */) {
                     query_point_raw[3] = starting_positions(transition, i); // Use i for column index matching current_controlled_servos loop
                 } else if (query_build_ok) {
                     Error("Starting position index out of bounds for transition " + std::to_string(transition) + ", servo index " + std::to_string(i));
                     query_build_ok = false;
                 }
                 // --- End raw query vector construction ---

                 if (!query_build_ok) {
                      Error("Failed to construct NN query vector due to missing data or index issues for servo " + std::to_string(servo_idx));
                      continue;
                 }


                 // --- Standardize the query vector ---
                 std::vector<float> query_point_standardized(num_features);
                 for (size_t feat_idx = 0; feat_idx < num_features; ++feat_idx) {
                      float mean = current_point_cloud_ptr->feature_means[feat_idx];
                      float std = current_point_cloud_ptr->feature_stds[feat_idx];
                      // Avoid division by zero
                      if (std != 0.0f) {
                          query_point_standardized[feat_idx] = (query_point_raw[feat_idx] - mean) / std;
                      } else {
                           // If std dev is zero, the standardized value should also be zero (as raw must equal mean)
                           query_point_standardized[feat_idx] = 0.0f;
                      }
                 }
                 // --- End standardization ---


                 // Perform the nearest neighbor search using the k-d tree
                 // <<< EDIT 5: Call updated FindNearestNeighborCurrent >>>
                 estimated_current = FindNearestNeighborCurrent(
                     query_point_standardized, // Pass standardized query
                     expected_num_features,    // Pass feature count (now fixed at 4)
                     *current_index_ptr,
                     *current_point_cloud_ptr
                 );
                 // <<< END EDIT 5 >>>

            }
            // <<< END EDIT 7 >>>
            else
            {
                Warning("Unsupported model type: " + model_name + " - using present current");
                estimated_current = present_current(servo_idx); // Use present as fallback
            }

            // Apply current limits and set output
            // Use abs() for estimated_current as current should likely be positive magnitude
            current_output(servo_idx) = std::min(std::abs(estimated_current), static_cast<double>(limit));

             // Store the *raw* model prediction before limiting for analysis later (if needed)
             // Overwrite the value in model_prediction which might have been set by ANN logic
             model_prediction(servo_idx) = estimated_current;
        }

        return current_output;
    }

    void StartPythonProcess()
    {
        std::string directory = __FILE__;
        // Get directory path just like you already do
        directory = directory.substr(0, directory.find_last_of("/"));
        std::string envPath = directory + "/.tensorflow_venv/bin/python3";
        std::string pythonPath = directory + "/ANN_prediction.py";

        child_pid = fork();
        if (child_pid < 0) {
             Error("Failed to fork process for Python script");
             return;
        }
        if (child_pid == 0) {
             // Child process
             Debug("StartPythonProcess (Child): Attempting to execl...");
             // This line fails because envPath is wrong!
             execl(envPath.c_str(), envPath.c_str(), pythonPath.c_str(), (char *)nullptr);
             // If exec returns, then an error occurred.
             Error("execl failed: " + std::string(strerror(errno)));
             exit(1);
        }
        // Parent process: continue execution and store child_pid for later cleanup.
    }

    void Init()
    {
        //IO
        Bind(present_current, "PresentCurrent");
        Bind(present_position, "PresentPosition");
        Bind(gyro, "GyroData");
        Bind(accel, "AccelData");
        Bind(eulerAngles, "EulerAngles");
        Bind(goal_current, "GoalCurrent");
        Bind(num_transitions, "NumberTransitions");
        Bind(goal_position_out, "GoalPositionOut");
        Bind(model_prediction, "ModelPrediction");
        Bind(model_prediction_start, "ModelPredictionStart");
        

        //parameters
        Bind(min_limits, "MinLimits");
        Bind(max_limits, "MaxLimits");
        Bind(robotType, "RobotType");
        Bind(prediction_model, "CurrentPrediction");
        Bind(current_increment_parameter, "CurrentIncrement");
        Bind(log_deviance_data, "DevianceLogging");
        current_increment = current_increment_parameter.as_int();

        std::string scriptPath = __FILE__;
        
       
        

        number_transitions = num_transitions.as_int()+1; // Add one to the number of transitions to include the ending position
        position_transitions.set_name("PositionTransitions");
        position_transitions = RandomisePositions(number_transitions, min_limits, max_limits, robotType.as_string());
        // position_transitions = {
        //     {180, 225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 218, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {126, 207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {183, 225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 218, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {126, 207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {183, 225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 218, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {126, 207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {183, 225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 218, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {126, 207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {183, 225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 218, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {126, 207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {183, 225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 218, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {126, 207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

        // position_transitions = {
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {180, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //     {125, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
            
            
            
    


            

        goal_position_out.copy(position_transitions[0]);

        goal_current.set(starting_current);
        max_present_current.set_name("MaxPresentCurrent");
        max_present_current.copy(present_current);
        min_moving_current.set_name("MinMovingCurrent");
        min_moving_current.copy(present_current);
        min_moving_current.set(1000);

        if (present_position.size() == 0)
        {
            Error("present_position is empty. Ensure that the input is connected and has valid dimensions.");
            return;
        }

        approximating_goal.set_name("ApproximatingGoal");
        approximating_goal.copy(present_position);
        approximating_goal.set(0);
        transition_start_time.set_name("StartedTransitionTime");
        transition_start_time.copy(approximating_goal);
        transition_duration.set_name("TransitionDuration");
        transition_duration.copy(approximating_goal);

                
        servo_names = {"NeckTilt", "NeckPan", "LeftEye", "RightEye", "LeftPupil", "RightPupil", "LeftArmJoint1", "LeftArmJoint2", "LeftArmJoint3", "LeftArmJoint4", "LeftArmJoint5", "LeftHand", "RightArmJoint1", "RightArmJoint2", "RightArmJoint3", "RightArmJoint4", "RightArmJoint5", "RightHand", "Body"};
        

        if(robotType.as_string() == "Torso"){
            current_controlled_servos.set_name("CurrentControlledServosTorso");
            current_controlled_servos = {0, 1};
            overshot_goal_temp = {false, false};
            reached_goal = {0, 0};
            
        }
        else if(robotType.as_string() == "FullBody"){
            current_controlled_servos.set_name("CurrentControlledServosFullBody");
            current_controlled_servos = {0, 1, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18};
            overshot_goal_temp = {false, false, false, false, false, false, false, false, false, false, false, false, false};
            reached_goal = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }
        else{
            Error("Robot type not recognized");
        }
        
        if (prediction_model.as_string() == "Linear" || prediction_model.as_string() == "Quadratic") {
        // go up in the directory to get to the folder containing the coefficients.json file
        std::string coefficientsPath = scriptPath.substr(0, scriptPath.find_last_of("/"));
        coefficientsPath = coefficientsPath.substr(0, coefficientsPath.find_last_of("/"));
        coefficientsPath = coefficientsPath + "/CurrentPositionMapping/models/coefficients.json";
            current_coefficients.load_json(coefficientsPath);

            coeffcient_matrix = CreateCoefficientsMatrix(current_coefficients, current_controlled_servos, prediction_model.as_string(), servo_names);
            
        }
        
        reached_goal.set_name("ReachedGoal");
        

        overshot_goal.set_name("OvershotGoal");
        unique_id = GenerateRandomNumber(0, 1000000);

    

        // Initialize new matrices for tracking prediction data
        initial_currents = matrix(number_transitions, current_controlled_servos.size());
        current_differences = matrix(number_transitions, current_controlled_servos.size());
        initial_currents.set(0);
        current_differences.set(0);

        // New: Initialize matrices to track the model's predicted values and prediction errors
        predicted_goal_current.set_name("PredictedGoalCurrent");
        predicted_goal_current = matrix(number_transitions, current_controlled_servos.size());
        predicted_goal_current.set(0);

        prediction_error.set_name("PredictionError");
        prediction_error = matrix(number_transitions, current_controlled_servos.size());
        prediction_error.set(0);

        model_prediction.set_name("ModelPrediction");
        model_prediction.copy(present_current);
        model_prediction.set(0);
        
        // Initialize the model_prediction_start matrix to store starting currents
        model_prediction_start.set_name("ModelPredictionStart");
        model_prediction_start.copy(present_current);
        model_prediction_start.set(0);
        
        if (std::string(prediction_model) == "ANN") {
            Debug("Attempting to create shared memory at " + std::string(SHM_NAME));
            
            // First, ensure any existing shared memory is removed
            shm_unlink(SHM_NAME);
            
            // Create shared memory
            shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
            if (shm_fd == -1) {
                Error("Failed to create shared memory: " + std::string(strerror(errno)) + " (errno: " + std::to_string(errno) + ")");
                return;
            }
            Debug("Successfully created shared memory. File descriptor: " + std::to_string(shm_fd));

            // Calculate size
            size_t required_size = sizeof(SharedData);
            Debug("Attempting to set shared memory size to " + std::to_string(required_size) + " bytes\n");
            
            // Set the size of shared memory
            if (ftruncate(shm_fd, required_size) == -1) {
                Error("Failed to set shared memory size (" + std::to_string(required_size) + " bytes): " + strerror(errno) + " (errno: " + std::to_string(errno) + ")");
                close(shm_fd);
                shm_unlink(SHM_NAME);
                return;
            }

            // Map the shared memory
            void* ptr = mmap(nullptr, required_size, PROT_READ | PROT_WRITE, 
                           MAP_SHARED, shm_fd, 0);
            if (ptr == MAP_FAILED) {
                Error("Failed to map shared memory: " + std::string(strerror(errno)));
                close(shm_fd);
                shm_unlink(SHM_NAME);
                return;
            }

            // Initialize the shared memory
            shared_data = static_cast<SharedData*>(ptr);
            memset(shared_data, 0, required_size);
            shared_data->new_data = false;

            
            Debug("Shared memory initialized:\n");
            Debug("- Required size: " + std::to_string(required_size) + " bytes\n");
            Debug("- SharedData size: " + std::to_string(sizeof(SharedData)) + " bytes\n");

            // Start Python process
            StartPythonProcess();
            Sleep(5); // Wait for the Python process to start
        }
        else if (std::string(prediction_model) == "NearestNeighbor") {
            Debug("NearestNeighbor model selected. Loading CSV data and building k-d trees.");
            if (robotType.as_string() != "Torso") {
                 Error("NearestNeighbor model currently only supports 'Torso' robotType.");
                 return;
            }

            // Base path for data files
            std::string dataBasePath = scriptPath.substr(0, scriptPath.find_last_of("/"));
            dataBasePath = dataBasePath.substr(0, dataBasePath.find_last_of("/"));
            dataBasePath = dataBasePath + "/CurrentPositionMapping/data/";

            bool tilt_load_success = false;
            bool pan_load_success = false;

            // --- Load Data and Build Index for Tilt ---
            // <<< EDIT 2: Use standardized CSV and mean/std JSON paths >>>
            std::string tilt_data_path = dataBasePath + "tilt_filtered_data_position_control_standardised.csv";
            std::string tilt_mean_std_path = dataBasePath + "tilt_filtered_data_position_control_mean_std.json";
            // <<< END EDIT 2 >>>
            Debug("Loading Tilt NN data from: " + tilt_data_path + " (using means/stds from " + tilt_mean_std_path + ")");

            // <<< EDIT 2: Call modified LoadCSVData >>>
            tilt_load_success = LoadCSVData(tilt_data_path, tilt_mean_std_path, nn_tilt_point_cloud, "Tilt");
            // <<< END EDIT 2 >>>

            if(tilt_load_success) {
                 Debug("Building k-d tree index for Tilt data (" + std::to_string(nn_tilt_point_cloud.pts.size()) + " points, " + std::to_string(nn_tilt_point_cloud.num_features) + " features)...");
                 // <<< EDIT 2: Construct k-d tree index passing dynamic dimension >>>
                 // Ensure num_features is > 0 before constructing
                 if (nn_tilt_point_cloud.num_features > 0) {
                     nn_tilt_index = std::make_unique<my_kd_tree_t>(
                         nn_tilt_point_cloud.num_features, // Pass dynamic dimension
                         nn_tilt_point_cloud,
                         nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max_leaf */)
                     );
                     nn_tilt_index->buildIndex();
                     Debug("Tilt k-d tree built.");
                 } else {
                     Error("Tilt PointCloud has 0 features after loading. Cannot build k-d tree.");
                     tilt_load_success = false; // Mark as failed if no features
                 }
                 // <<< END EDIT 2 >>>
            } else {
                Error("Failed to load Tilt NN data. Nearest neighbor search for tilt will not work.");
            }


             // --- Load Data and Build Index for Pan ---
             // <<< EDIT 2: Use standardized CSV and mean/std JSON paths >>>
             std::string pan_data_path = dataBasePath + "pan_filtered_data_position_control_standardised.csv";
             std::string pan_mean_std_path = dataBasePath + "pan_filtered_data_position_control_mean_std.json";
             // <<< END EDIT 2 >>>
             Debug("Loading Pan NN data from: " + pan_data_path + " (using means/stds from " + pan_mean_std_path + ")");

             // <<< EDIT 2: Call modified LoadCSVData >>>
             pan_load_success = LoadCSVData(pan_data_path, pan_mean_std_path, nn_pan_point_cloud, "Pan");
             // <<< END EDIT 2 >>>

             if (pan_load_success) {
                 Debug("Building k-d tree index for Pan data (" + std::to_string(nn_pan_point_cloud.pts.size()) + " points, " + std::to_string(nn_pan_point_cloud.num_features) + " features)...");
                 // <<< EDIT 2: Construct k-d tree index passing dynamic dimension >>>
                 if (nn_pan_point_cloud.num_features > 0) {
                     nn_pan_index = std::make_unique<my_kd_tree_t>(
                         nn_pan_point_cloud.num_features, // Pass dynamic dimension
                         nn_pan_point_cloud,
                         nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max_leaf */)
                     );
                     nn_pan_index->buildIndex();
                     Debug("Pan k-d tree built.");
                 } else {
                      Error("Pan PointCloud has 0 features after loading. Cannot build k-d tree.");
                      pan_load_success = false; // Mark as failed if no features
                 }
                 // <<< END EDIT 2 >>>
             } else {
                 Error("Failed to load Pan NN data. Nearest neighbor search for pan will not work.");
             }
             // Check if both failed?
             if (!tilt_load_success && !pan_load_success) {
                 Error("Failed to load data for both Tilt and Pan. NN model unusable.");
                 return; // Definitely stop if both failed
             }
             // <<< EDIT 2: Add consistency check for feature count and names >>>
             if (tilt_load_success && pan_load_success) {
                 if (nn_tilt_point_cloud.num_features != nn_pan_point_cloud.num_features) {
                     Error("Mismatch in number of features between Tilt (" + std::to_string(nn_tilt_point_cloud.num_features) +
                           ") and Pan (" + std::to_string(nn_pan_point_cloud.num_features) + ") data.");
                     return; // Stop if dimensions don't match
                 }
                 if (nn_tilt_point_cloud.feature_names != nn_pan_point_cloud.feature_names) {
                      Warning("Feature names differ between Tilt and Pan datasets. Ensure this is intended.");
                      // Optionally print differences for debugging
                 }
                 Debug("NN Feature count: " + std::to_string(nn_tilt_point_cloud.num_features));
             } else if (tilt_load_success) {
                 Debug("NN Feature count (Tilt only): " + std::to_string(nn_tilt_point_cloud.num_features));
             } else if (pan_load_success) {
                 Debug("NN Feature count (Pan only): " + std::to_string(nn_pan_point_cloud.num_features));
             }
             // <<< END EDIT 2 >>>
        }

  
        timeout_occurred.set_name("TimeoutOccurred");
        timeout_occurred = matrix(number_transitions, current_controlled_servos.size());
        timeout_occurred.set(0);

        // Initialize transition_start_time as a 2D matrix
        transition_start_time.set_name("StartedTransitionTime");
        transition_start_time = matrix(number_transitions, current_controlled_servos.size());
        transition_start_time.set(0);

        starting_positions = matrix( number_transitions, current_controlled_servos.size());
        starting_positions.set(0);

        // <<< EDIT 2: Initialize deviance logging if enabled >>>

        if (log_deviance_data) {
            try {
                std::string scriptPath = __FILE__;
                std::string scriptDirectory = scriptPath.substr(0, scriptPath.find_last_of("/\\"));
                std::string resultsDir = scriptDirectory + "/results/deviance_logs/" + std::string(prediction_model);

                // Create directory if it doesn't exist
                if (!std::filesystem::exists(resultsDir)) {
                    if (!std::filesystem::create_directories(resultsDir)) {
                         Error("Failed to create results directory: " + resultsDir);
                         log_deviance_data = false; // Disable logging if directory fails
                    }
                }

                if (log_deviance_data) {
                    // Get current time for unique filename
                     auto now = std::chrono::system_clock::now();
                     auto now_c = std::chrono::system_clock::to_time_t(now);
                     std::stringstream ss_time;
                     ss_time << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M");
                     std::string time_stamp_str = ss_time.str();

                    std::string filepath = resultsDir + "/deviance_log_" + time_stamp_str + "position_control" + ".json";

                    deviance_log_file.open(filepath, std::ios::out | std::ios::trunc);
                    if (!deviance_log_file.is_open()) {
                        Error("Failed to open deviance log file: " + filepath);
                        log_deviance_data = false; // Disable logging on failure
                    } else {
                        Debug("Opened deviance log file: " + filepath);
                        // Initialize with a valid empty JSON array structure
                        deviance_log_file << "{\n\"deviance_data\" : [\n{ \n]}\n";
                        deviance_log_file.flush(); // Ensure it's written
                        is_first_deviance_log_entry = true; // This flag now means "is the *next logged entry* the first one in the array?"
                    }
                }
            } catch (const std::exception& e) {
                Error("Exception during deviance log initialization: " + std::string(e.what()));
                log_deviance_data = false;
                 if (deviance_log_file.is_open()) {
                    deviance_log_file.close();
                 }
            }
        }
        // <<< END EDIT 2 >>>
    }

    
    void Tick()
    {   
        //print tick
        Debug("Tick " + std::to_string(GetTick()));
        
        // First check if required inputs are connected
        if (!present_current.connected() || !present_position.connected()) {
            Error("Present current and present position must be connected");
            return;
        }

        // Ensure matrices are properly sized before operations
        if (goal_position_out.size() != position_transitions.cols()) {
            goal_position_out.resize(position_transitions.cols());
        }
        
        ReachedGoal(present_position, goal_position_out, reached_goal, position_margin);
        approximating_goal = ApproximatingGoal(present_position, goal_position_out, position_margin);

        // Skip processing on first tick, just initialize start time
        if (GetTick() <= 2) {
            transition_start_time[transition].set(GetNominalTime());
            for (int i = 0; i < current_controlled_servos.size(); i++) {
                starting_positions(transition, i) = present_position(current_controlled_servos(i));
            }
            return;
        }

        // Check if all servos reached their goals for current transition
        if (reached_goal.sum() == current_controlled_servos.size()) {
            // For each current controlled servo, record the final current difference and prediction error
            for (int i = 0; i < current_controlled_servos.size(); i++) {
                int servo_idx = current_controlled_servos(i);
                if (current_differences(transition, i) == 0) {
                    current_differences(transition, i) = present_current[servo_idx] - initial_currents(transition, i);
                    goal_current[servo_idx] = present_current[servo_idx];
                    
                }
            }
            PrintProgressBar(transition, number_transitions);
            transition++;
            
            
            // Save data after every 5th transition
            if (transition % 2 == 0) {
                SaveCurrentData();
                Debug("Saved data after transition " + std::to_string(transition));
            }
            
            if (transition < number_transitions) {
                // Store the starting position for the new transition
                for (int i = 0; i < current_controlled_servos.size(); i++) {
                    int servo_idx = current_controlled_servos(i);
                    transition_start_time(transition, i) = GetNominalTime();
                    starting_positions(transition, i) = present_position(servo_idx);
                    reached_goal(servo_idx) = 0;
                    approximating_goal(servo_idx) = 0;
                }
                if (log_deviance_data) {
                    Sleep(0.5);
                }
                goal_position_out.copy(position_transitions[transition]);
                goal_current.copy(present_current);

                // Compute and store the predicted goal current from the model for this new transition
                // Only do this once per transition
                matrix all_predicted = SetGoalCurrent(present_current, current_increment, current_limit,
                                                      present_position, goal_position_out, position_margin,
                                                      coeffcient_matrix, current_coefficients, std::string(prediction_model), model_prediction);
                for (int i = 0; i < current_controlled_servos.size(); i++) {
                    int servo_idx = current_controlled_servos(i);
                    predicted_goal_current(transition, i) = all_predicted(servo_idx);
                }
            } else {
                // Still save at the end if not already saved
                if (transition % 5 != 0) {
                    SaveCurrentData();
                }
                Notify(msg_terminate, "Transition finished");
                return;
            }
        }

        // Process ANN predictions if using that model
        if (std::string(prediction_model) == "ANN" && gyro.connected() && accel.connected() && eulerAngles.connected()) {
          
            int idx = 0;
            
            // First write all gyro data (3 values)
            for (int i = 0; i < gyro.size(); i++) {
                shared_data->data[idx++] = gyro[i];
            }
            
            // Then write all accel data (3 values)
            for (int i = 0; i < accel.size(); i++) {
                shared_data->data[idx++] = accel[i];
            }
            
            // Finally write all euler angle data (3 values)
            for (int i = 0; i < eulerAngles.size(); i++) {
                shared_data->data[idx++] = eulerAngles[i];
            }
            
            // For each servo, write all its data together
            for (int i = 0; i < current_controlled_servos.size(); i++) {
                int servo_idx = current_controlled_servos(i);
                
                // Current position
                shared_data->data[idx++] = present_position[servo_idx];
                
                // Distance to goal -- potential bug. Changed from absolute value 
                shared_data->data[idx++] = (float)goal_position_out[servo_idx] - (float)present_position[servo_idx];

                //Distance to start position
                shared_data->data[idx++] = (float)starting_positions(transition, i) - (float)present_position[servo_idx];
                
                // Goal position
                shared_data->data[idx++] = goal_position_out[servo_idx];
                
                // Starting position
                shared_data->data[idx++] = starting_positions(transition, i);

            }

            num_input_data = idx;
            // Debug, display input data
            std::string debug_input = "Input data sent to ANN: ";
            for (int i = 0; i < idx; i++) {
                debug_input += std::to_string(i) + "=" + std::to_string(shared_data->data[i]) + " ";
            }
            Debug(debug_input);

            // Set the new_data flag to true BEFORE memory fence to ensure Python sees it
            shared_data->new_data = true;
            
            // Memory fence to ensure all writes are visible to other processes
            std::atomic_thread_fence(std::memory_order_release);
            
            // Wait for Python to process (with timeout)
            int timeout_ms = 5  ;  // 5ms timeout
            auto start = std::chrono::steady_clock::now();
            bool got_response = false;
            
            while (!got_response) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
                
                if (elapsed > timeout_ms) {
                    Warning("Timeout waiting for ANN prediction. Elapsed time: " + std::to_string(elapsed) + "ms. Tick: " + std::to_string(GetTick()));
                    break;
                }

                // Memory fence to ensure we see the most recent value of the flag
                std::atomic_thread_fence(std::memory_order_acquire);
                
                // New data flag is set to false by Python when it has processed the data
                if (!shared_data->new_data) {
                    // Read predictions and update model_prediction dynamically based on number of servos
                    std::string debug_msg = "Received predictions: ";
                    
                    // Debug output showing all values in shared data
                    std::string all_data = "All shared data: ";
                    for (int i = 0; i < FLOAT_COUNT; i++) {
                        all_data += std::to_string(i) + "=" + std::to_string(shared_data->data[i]) + " ";
                    }
                    
                    
                    // Calculate base offset for predictions (after all input data)
                    int prediction_base_offset = num_input_data;  // This is where predictions start (after 17 input values)

                    // Dynamically handle predictions for any number of servos
                    for (int i = 0; i < current_controlled_servos.size(); i++) {
                        int servo_idx = current_controlled_servos(i);
                        // Each servo has 2 predictions (regular and start current)
                        int regular_offset = prediction_base_offset + (i );
                        
                        
                        // Update predictions
                        model_prediction[servo_idx] = shared_data->data[regular_offset];
                        
                        
                        // Build debug message
                        if (i > 0) debug_msg += ", ";
                        debug_msg += std::string(servo_names[servo_idx]) + 
                                    "=(regular=" + std::to_string(model_prediction[servo_idx]) + 
                                    ")";
                    }
                    std::cout << "Tick: " << GetTick() << std::endl;
                    present_current.print();
                    model_prediction.print();

                    got_response = true;
                    Debug(debug_msg);
                }

                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }

            if (!got_response) {
                Warning("Using previous ANN predictions due to timeout");
            }
        }

       
        
        // Process servo movements and timeouts
        for (int i = 0; i < current_controlled_servos.size(); i++) {
            int servo_idx = current_controlled_servos(i);
            double current_time = GetNominalTime();
            bool timeout = current_time - transition_start_time(transition, servo_idx) > TIMEOUT_DURATION;

            // If servo hasn't reached goal yet
            if (reached_goal(servo_idx) == 0) {
                if (!timeout) {
                    // Use ANN output directly as goal current if using ANN model
                    if (std::string(prediction_model) == "ANN") {

                        // Check if the servo is still close to its starting position for this transition
                        float start_pos = starting_positions(transition, i);
                        float current_pos = present_position(servo_idx);
                        
                        
                        goal_current(servo_idx) = std::min<float>(abs(model_prediction[servo_idx]), (float)current_limit);
                        
                        // Add debug to verify the correct current is being used
                        Debug("Using REGULAR current (moved from start pos) for " + std::string(servo_names[servo_idx]) + 
                                ": " + std::to_string(model_prediction[servo_idx]));
                    
                    } else {
                        // For other models, use the existing prediction method
                        goal_current.copy(SetGoalCurrent(present_current, current_increment, current_limit,
                                                   present_position, goal_position_out, position_margin,
                                                   coeffcient_matrix, current_coefficients, std::string(prediction_model), model_prediction));
                    }
                } else {
                    // Handle timeout case
                    if (timeout_occurred(transition, i) == 0) {
                        timeout_occurred(transition, i) = 1;
                        Warning("Timeout started for servo " + std::string(servo_names[servo_idx]) + 
                               " in transition " + std::to_string(transition));
                 
                    }
                    if (approximating_goal(servo_idx) == 0) {
                        goal_current(servo_idx) = std::min(goal_current[servo_idx] + current_increment, (float)current_limit);
                    }

                    // Calculate prediction error if we've reached at least 10% of the distance to goal
                    // and we haven't already calculated it for this servo in this transition
                    if (timeout_occurred(transition, i) == 1 && prediction_error(transition, i) == 0) {
                        // Calculate how far we've moved toward the goal
                        float start_pos = starting_positions(transition, i);
                        float goal_pos = position_transitions(transition, servo_idx);
                        float current_pos = present_position(servo_idx);
                        float total_distance = abs(goal_pos - start_pos);
                        float traveled_distance = abs(current_pos - start_pos);
                        
                        // Check if we've traveled at least 1% of the distance to goal
                        if (total_distance > 0 && traveled_distance >= 0.01 * total_distance) {
                            // Calculate prediction error (absolute difference between predicted and actual current)
                            float predicted = predicted_goal_current(transition, i);
                            float actual = present_current(servo_idx);
                            prediction_error(transition, i) = abs(predicted - actual);
                            
                            Debug("Prediction error calculated for " + std::string(servo_names[servo_idx]) + 
                                  " in transition " + std::to_string(transition) + 
                                  ": " + std::to_string(prediction_error(transition, i)));
                        }
                    }
                }
                
                
            }

            
        }

        if (GetTick() >= 2 && predicted_goal_current.sum() == 0) {
            matrix all_predicted = SetGoalCurrent(present_current, current_increment, current_limit,
                                                  present_position, goal_position_out, position_margin,
                                                  coeffcient_matrix, current_coefficients, std::string(prediction_model), model_prediction);
            
            for (int i = 0; i < current_controlled_servos.size(); i++) {
                int servo_idx = current_controlled_servos(i);
                predicted_goal_current(transition, i) = all_predicted(servo_idx);
            }
        }
        
        // <<< EDIT 4: Call LogDevianceData if enabled >>>
        // Log data *before* potentially changing state in this tick (like transition increment)
        if (log_deviance_data && GetTick() > 2) { // Start logging after initial ticks
            goal_current[0] = 2000;
            goal_current[1] = 500;
            
            
            LogDevianceData();
        }
        // <<< END EDIT 4 >>>
    }

    void SaveCurrentData()
    {
        try
        {
            std::string scriptPath = __FILE__;
            std::string scriptDirectory = scriptPath.substr(0, scriptPath.find_last_of("/\\"));
            
            // Use the same filename for all saves (remove the transition number)
            std::string filepath = scriptDirectory + "/results/" + std::string(prediction_model);

            //create directory if it doesn't exist
            std::string directory = scriptDirectory + "/results/" + std::string(prediction_model);
            if (!std::filesystem::exists(directory)) {
                std::filesystem::create_directory(directory);
            }

            filepath = directory + "/current_data" + ".json";
           
            std::ofstream file(filepath);
            file << "{\n";
            file << "  \"transitions\": [\n";

            file << "    {\n";

            for (int i = 0; i < current_controlled_servos.size(); i++)
            {
                file << "      \"" << servo_names[current_controlled_servos(i)] << "\": {\n";

                // Log the starting positions
                file << "        \"starting_positions\": [";
                for (int t = 0; t < transition; t++) {
                    file << starting_positions(t, i);
                    if (t < transition - 1) file << ", ";
                }
                file << "],\n";

                file << "        \"goal_positions\": [";
                for (int t = 0; t < transition; t++) {
                    file << position_transitions(t, current_controlled_servos(i));
                    if (t < transition - 1) file << ", ";
                }
                file << "],\n";

                file << "        \"predicted_goal_currents\": [";
                for (int t = 0; t < transition; t++) {
                    file << predicted_goal_current(t, i);
                    if (t < transition - 1) file << ", ";
                }
                file << "],\n";

                file << "        \"current_differences\": [";
                for (int t = 0; t < transition; t++) {
                    file << current_differences(i, t);
                    if (t < transition - 1) file << ", ";
                }
                file << "],\n";

                // Add timeout statistics
                file << "        \"timeout_occurred\": [";
                for (int t = 0; t < transition; t++) {
                    file << timeout_occurred(t, i);
                    if (t < transition - 1) file << ", ";
                }
                file << "],\n";

                // Log the prediction errors
                file << "        \"prediction_errors\": [";
                for (int t = 0; t < transition; t++) {
                    file << prediction_error(t, i);
                    if (t < transition - 1) file << ", ";
                }
                file << "]\n";

                file << "      }";
                if (i < current_controlled_servos.size() - 1) file << ",";
                file << "\n";
            }
            
            file << "    }";
            file << "\n";
            
            file << "  ]\n";
            file << "}\n";
            file.close();
            
            Debug("Data saved to " + filepath + " (progress: " + std::to_string(transition) + "/" + 
                  std::to_string(number_transitions) + " transitions)");
        }
        catch (const std::exception &e)
        {
            Error("Failed to save current data: " + std::string(e.what()));
        }
    }

    void SignalShutdownToANN()
    {
        if (shared_data) {
            // Set shutdown flag (e.g., to 1) at the last byte of shared memory
            char shutdown_val = 1;
            // The shutdown flag is at offset MEM_SIZE - 1
            memcpy(((char*)shared_data) + MEM_SIZE - 1, &shutdown_val, sizeof(shutdown_val));
            // Ensure the write completes before proceeding
            std::atomic_thread_fence(std::memory_order_release);
        }
    }

    ~TestMapping()
    {
        if (std::string(prediction_model) == "ANN") {
            // Set shutdown flag and wait for Python process to exit gracefully
            if (shared_data != nullptr) {
                // First signal shutdown through flag
                shared_data->new_data = false;  // Reset data flag
                
                // Set shutdown flag (second boolean in shared memory)
                char* flags_ptr = (char*)shared_data + (FLOAT_COUNT * sizeof(float));
                *(flags_ptr + 1) = 1;  // Set shutdown flag to true
                
                // Ensure write completes
                std::atomic_thread_fence(std::memory_order_release);
                
                // Wait for Python to notice shutdown flag before cleaning up memory
                sleep(1);
            }

            // Terminate child process if it exists
            if (child_pid > 0) {
                // Send SIGTERM first for graceful shutdown
                kill(child_pid, SIGTERM);
                
                // Wait for a short period
                struct timespec ts;
                ts.tv_sec = 0;
                ts.tv_nsec = 500000000; // 500ms
                nanosleep(&ts, NULL);
                
                // Check if process still exists and force kill if needed
                if (waitpid(child_pid, nullptr, WNOHANG) == 0) {
                    kill(child_pid, SIGKILL);
                    waitpid(child_pid, nullptr, 0);
                }
            }

            // Clean up shared memory resources in correct order
            if (shared_data != nullptr) {
                munmap(shared_data, MEM_SIZE);  // Use MEM_SIZE consistently
                shared_data = nullptr;
            }
            
            if (shm_fd != -1) {
                close(shm_fd);
                shm_fd = -1;
            }
            
            // Always unlink at the end
            shm_unlink(SHM_NAME);

            // As a fallback, kill any lingering processes
            system("lsof -ti:8000 | xargs kill -9 2>/dev/null || true");
        }

        // Finalize and close deviance log file
        if (log_deviance_data && deviance_log_file.is_open()) {
            // Write any remaining data from the buffer
            if (!deviance_log_stream.str().empty()) { // Check if the buffer has content
                deviance_log_file.seekp(-7L, std::ios::end); // Seek to before the closing "\\n  ]\\n}\\n"

                if (!is_first_deviance_log_entry) { // If not the first ever entry, add a comma
                    deviance_log_file << ",\n";
                }
                
                deviance_log_file << deviance_log_stream.str(); // Write the remaining buffered objects
                                
                deviance_log_file << "\n  ]\n}\n"; // Add the final closing structure
                deviance_log_file.flush(); // Ensure it's written
            }
            // If deviance_log_stream was empty, the file was already correctly closed 
            // by the last flush in LogDevianceData, or it's still in its initial valid empty state from Init.
            
            deviance_log_file.close();
            Debug("Closed deviance log file.");
        }
    }

    // <<< EDIT 6: Add function signature >>>
    void LogDevianceData();
    // <<< END EDIT 6 >>>

    // <<< EDIT 3: Implement helper functions >>>
    // Helper function to load CSV data and populate the PointCloud structure
    // Returns true on success, false on failure.
    bool LoadCSVData(const std::string& filepath, const std::string& mean_std_filepath, PointCloud& point_cloud_out, std::string servo);
    double FindNearestNeighborCurrent(const std::vector<float>& query_point_standardized, const size_t num_features, const std::unique_ptr<my_kd_tree_t>& index_ptr, const PointCloud& point_cloud);
    // <<< END EDIT 3 >>>
};

// <<< EDIT 3: Implement helper functions >>>
// Helper function to load CSV data and populate the PointCloud structure
// Returns true on success, false on failure.
bool TestMapping::LoadCSVData(const std::string& filepath, const std::string& mean_std_filepath, PointCloud& point_cloud_out, std::string servo) {
    point_cloud_out.pts.clear();
    point_cloud_out.target_currents.clear();
    point_cloud_out.feature_names.clear();
    point_cloud_out.feature_means.clear();
    point_cloud_out.feature_stds.clear();
    point_cloud_out.num_features = 0;
    point_cloud_out.normalization_enabled = false;
    // --- 1. Load Mean/Std JSON ---
    nlohmann::json mean_std_data;
    std::ifstream mean_std_file(mean_std_filepath);
    if (!mean_std_file.is_open()) {
        Error("Failed to open mean/std JSON file: " + mean_std_filepath);
        return false;
    }
    try {
        mean_std_file >> mean_std_data;
        mean_std_file.close();
    } catch (json::parse_error& e) {
        Error("Failed to parse mean/std JSON file: " + mean_std_filepath + " - " + e.what());
        if(mean_std_file.is_open()) mean_std_file.close();
        return false;
    }

    // --- 2. Open and Read CSV Header ---
    std::ifstream file(filepath);
    if (!file.is_open()) {
        Error("Failed to open CSV file: " + filepath);
        return false;
    }

    std::string line;
    std::vector<std::string> headers;
    std::string target_label = "";
    int target_col_index = -1;
    std::vector<int> feature_indices_in_csv; // Stores CSV column index for each feature *in order*
    std::vector<std::string> ordered_feature_names; // Stores feature names *in order*
    // <<< EDIT: Define the desired features and their fixed order >>>
    std::vector<std::string> desired_feature_base_names = {"GyroX", "GyroY", "GyroZ", "AngleX", "AngleY", "AngleZ", "Position", "DistToGoal", "GoalPosition", "StartPosition"};
    std::vector<std::string> desired_feature_full_names(desired_feature_base_names.size());
    for(size_t i = 0; i < desired_feature_base_names.size(); ++i) {
        desired_feature_full_names[i] = servo + desired_feature_base_names[i];
    }
    std::map<std::string, int> desired_feature_map; // Map full name to its fixed index (0-3)
    for(size_t i = 0; i < desired_feature_full_names.size(); ++i) {
        desired_feature_map[desired_feature_full_names[i]] = i;
    }
    // Vectors to store loaded feature info temporarily, ordered by desired_feature_map index
    std::vector<int> temp_feature_indices_in_csv(desired_feature_full_names.size(), -1);
    std::vector<std::string> temp_ordered_feature_names(desired_feature_full_names.size(), "");
    // <<< END EDIT >>>

    // Read header line
    if (std::getline(file, line)) {
        // Remove potential UTF-8 BOM and trailing \r
        if (line.size() >= 3 && line[0] == (char)0xEF && line[1] == (char)0xBB && line[2] == (char)0xBF) {
            line = line.substr(3);
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        std::stringstream ss(line);
        std::string header;
        int current_col = 0;
        while (std::getline(ss, header, ',')) {
            // Trim whitespace
            header.erase(0, header.find_first_not_of(" \t\n\r\f\v"));
            header.erase(header.find_last_not_of(" \t\n\r\f\v") + 1);
            headers.push_back(header);

            // Check if it's a target or a feature
            if (header.find(servo + "Current") != std::string::npos) {
                if (target_col_index != -1) {
                    Error("Multiple columns containing 'Current' found in header of " + filepath + ". Ambiguous target.");
                    return false;
                }
                target_label = header;
                target_col_index = current_col;
            } else {
                // <<< EDIT: Check if the header matches one of the desired features >>>
                if (desired_feature_map.count(header)) {
                    // Check if this feature exists in the mean/std data AND is not "UniqueId"
                    if (mean_std_data.contains(header)) {
                        // It's a desired feature we expect, store its info based on its fixed index
                        int fixed_index = desired_feature_map[header];
                        temp_feature_indices_in_csv[fixed_index] = current_col;
                        temp_ordered_feature_names[fixed_index] = header;
                    } else {
                         Warning("Desired feature header '" + header + "' in CSV " + filepath + " not found in mean/std JSON " + mean_std_filepath + ". Skipping this feature.");
                    }
                // <<< END EDIT >>>
                } else {
                    // Print warning only if column contain servo name but was skipped
                    if (header.find(servo) != std::string::npos) {
                         Warning("Header '" + header + "' in CSV " + filepath + " not found in mean/std JSON " + mean_std_filepath + " or is excluded. Skipping this column.");
                    }
                }
            }
            current_col++;
        }
       
    } else {
        Error("CSV file is empty or could not read header: " + filepath);
        return false;
    }

    // Validate headers and mean/std consistency
    if (target_col_index == -1) {
        Error("No target column (containing 'Current') found in CSV header: " + filepath);
        return false;
    }
  

    // <<< EDIT: Populate final feature lists based on successfully found desired features >>>
    feature_indices_in_csv.clear();
    ordered_feature_names.clear();
    for(size_t i = 0; i < desired_feature_full_names.size(); ++i) {
        if (temp_feature_indices_in_csv[i] != -1 && !temp_ordered_feature_names[i].empty()) {
            feature_indices_in_csv.push_back(temp_feature_indices_in_csv[i]);
            ordered_feature_names.push_back(temp_ordered_feature_names[i]);
        } else {
             Warning("Desired feature '" + desired_feature_full_names[i] + "' was not found or loaded successfully from CSV/JSON. It will be excluded.");
             // We might need to decide if this is an error condition depending on requirements.
        }
    }
    // <<< END EDIT >>>

    // Populate PointCloud feature info and load means/stds in the correct order
    point_cloud_out.feature_names = ordered_feature_names;
    point_cloud_out.num_features = ordered_feature_names.size(); // Use the count of actually loaded features
    point_cloud_out.feature_means.resize(point_cloud_out.num_features);
    point_cloud_out.feature_stds.resize(point_cloud_out.num_features);

    bool mean_std_load_ok = true;
    for (size_t i = 0; i < point_cloud_out.num_features; ++i) {
        const std::string& name = point_cloud_out.feature_names[i];
        try {
            if (!mean_std_data[name].contains("mean") || !mean_std_data[name].contains("std")) {
                 Error("Mean or std missing for feature '" + name + "' in JSON file: " + mean_std_filepath);
                 mean_std_load_ok = false;
                 break;
            }
            point_cloud_out.feature_means[i] = mean_std_data[name]["mean"].get<float>();
            point_cloud_out.feature_stds[i] = mean_std_data[name]["std"].get<float>();
            // Check for zero std dev (potential division by zero later)
            if (point_cloud_out.feature_stds[i] == 0.0f) {
                Warning("Standard deviation is zero for feature '" + name + "' in " + mean_std_filepath);
                // Decide how to handle: Keep it 0? Set to 1? For now, keep 0.
            }
        } catch (json::exception& e) {
            Error("Error accessing mean/std for feature '" + name + "' in JSON file: " + mean_std_filepath + " - " + e.what());
            mean_std_load_ok = false;
            break;
        }
    }

    if (!mean_std_load_ok) {
        return false; // Stop if mean/std loading failed
    }
    point_cloud_out.normalization_enabled = true; // Mark normalization data as loaded

    // --- 3. Read Data Rows ---
    while (std::getline(file, line)) {
        // Remove trailing \r and skip empty lines
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<float> all_row_values; // Stores all values from the CSV row temporarily
        while (std::getline(ss, cell, ',')) {
            try {
                 if (cell.empty()) {
                      Error("Empty cell encountered in CSV file: " + filepath);
                      return false; // Fail on empty cells
                 }
                all_row_values.push_back(std::stof(cell));
            } catch (const std::invalid_argument& e) {
                Error("Invalid numeric value '" + cell + "' in CSV file: " + filepath + " - " + e.what());
                return false;
            } catch (const std::out_of_range& e) {
                 Error("Numeric value out of range '" + cell + "' in CSV file: " + filepath + " - " + e.what());
                 return false;
            }
        }

        // Check if row size matches header size
        if (all_row_values.size() != headers.size()) {
            Warning("Row has different number of columns (" + std::to_string(all_row_values.size()) +
                    ") than header (" + std::to_string(headers.size()) + ") in CSV: " + filepath + ". Skipping row.");
            // Print header 
            std::string header_str = "Header: ";
            for(const auto& h : headers) header_str += h + " ";
            Warning(header_str);
            // Print all_row_values
            std::string values_str = "Values: ";
            for(const auto& v : all_row_values) values_str += std::to_string(v) + " ";
            Warning(values_str);
            continue;
        }

        // Extract feature values (already standardized in the CSV) and the target value
        std::vector<float> feature_values(point_cloud_out.num_features);
        for(size_t j = 0; j < point_cloud_out.num_features; ++j) {
            int csv_col_idx = feature_indices_in_csv[j];
            // Bounds check (should be okay due to previous checks)
            if (csv_col_idx < 0 || csv_col_idx >= all_row_values.size()) {
                 Error("Internal error: Invalid feature index " + std::to_string(csv_col_idx) + " while reading row.");
                 // Skip row or return false? Returning false for safety.
                 return false;
            }
            feature_values[j] = all_row_values[csv_col_idx];
        }

        // Extract target value
        float target_value = 0.0f;
        if (target_col_index >= 0 && target_col_index < all_row_values.size()) {
            target_value = all_row_values[target_col_index];
        } else {
             Error("Internal error: Invalid target index " + std::to_string(target_col_index) + " while reading row.");
             return false;
        }

        // Add data to PointCloud
        point_cloud_out.pts.push_back(feature_values);
        point_cloud_out.target_currents.push_back(target_value);
    }

    file.close();

    if (point_cloud_out.pts.empty()) {
        Warning("No valid data rows loaded into PointCloud from CSV: " + filepath);
        return false; // Treat as failure if no points loaded
    }

    Debug("Loaded " + std::to_string(point_cloud_out.pts.size()) + " data points (" +
          std::to_string(point_cloud_out.num_features) + " features) into PointCloud from " + filepath);
    Debug("Target column: '" + target_label + "' (index " + std::to_string(target_col_index) + ")");
    // Optional: Print feature names
    std::string features_str = "Features loaded: ";
    for(const auto& name : point_cloud_out.feature_names) features_str += name + " ";
    Debug(features_str);

    // <<< EDIT 8: Load target mean/std >>>
    // Find the target label identified earlier and load its mean/std
    if (target_col_index != -1 && !target_label.empty()) {
        if (mean_std_data.contains(target_label)) {
            try {
                if (!mean_std_data[target_label].contains("mean") || !mean_std_data[target_label].contains("std")) {
                     Error("Mean or std missing for target '" + target_label + "' in JSON file: " + mean_std_filepath);
                     // Don't set normalization_enabled to true if target stats are missing
                     point_cloud_out.normalization_enabled = false;
                     return false; // Cannot proceed without target stats for denormalization
                }
                point_cloud_out.target_mean = mean_std_data[target_label]["mean"].get<float>();
                point_cloud_out.target_std = mean_std_data[target_label]["std"].get<float>();

                // Check for zero target std dev
                if (point_cloud_out.target_std == 0.0f) {
                     Warning("Target standard deviation is zero for '" + target_label + "' in " + mean_std_filepath + ". Denormalization might be inaccurate.");
                     // Keep std as 0. The denormalized value will just be the mean.
                }
                 Debug("Loaded target normalization for '" + target_label + "': mean=" + std::to_string(point_cloud_out.target_mean) + ", std=" + std::to_string(point_cloud_out.target_std));

            } catch (json::exception& e) {
                 Error("Error accessing mean/std for target '" + target_label + "' in JSON file: " + mean_std_filepath + " - " + e.what());
                 point_cloud_out.normalization_enabled = false; // Mark as failed
                 return false;
            }
        } else {
            Error("Target label '" + target_label + "' found in CSV header but not in mean/std JSON: " + mean_std_filepath);
            point_cloud_out.normalization_enabled = false; // Mark as failed
            return false;
        }
    } else {
         Error("Could not identify target column or label during mean/std loading.");
         return false; // Should have been caught earlier, but safety check
    }
    // <<< END EDIT 8 >>>

    return true;
}

// Helper function to find the nearest neighbor using the k-d tree index
double TestMapping::FindNearestNeighborCurrent(
    const std::vector<float>& query_point_standardized, // Expect standardized query
    const size_t num_features, // Pass dimensionality explicitly
    const std::unique_ptr<my_kd_tree_t>& index_ptr, // Pass pointer to the index
    const PointCloud& point_cloud) // Pass associated point cloud for target lookup
{
    // Validate inputs
    if (!index_ptr) {
        Error("Invalid k-d tree index pointer passed to FindNearestNeighborCurrent.");
        return 0.0;
    }
     if (query_point_standardized.size() != num_features) {
         Error("Query point dimension (" + std::to_string(query_point_standardized.size()) +
               ") does not match expected feature count (" + std::to_string(num_features) + ").");
         return 0.0; // Return a default value or handle error
     }
     if (point_cloud.pts.empty() || point_cloud.num_features != num_features) {
          Error("Point cloud associated with k-d tree index is empty or has mismatched dimensions.");
          return 0.0;
     }


    // Perform k-NN search: find K nearest neighbors
    const size_t num_results_k = 1; // Use k=3
    std::vector<size_t> ret_indices(num_results_k); // Indices of the nearest neighbors
    std::vector<float> out_dist_sqr(num_results_k); // Squared distances to neighbors

    // Prepare result set
    // Use KNNResultSet: requires max_size (k), indices pointer, distances pointer
    nanoflann::KNNResultSet<float> resultSet(num_results_k);
    resultSet.init(&ret_indices[0], &out_dist_sqr[0]); // Provide pointers to first elements

    // Perform the search
    // query_point_standardized.data() returns a float* needed by findNeighbors
    index_ptr->findNeighbors(resultSet, query_point_standardized.data(), nanoflann::SearchParameters()); // Use default SearchParams

    // --- Calculate weighted average ---
    double weighted_sum_currents = 0.0;
    double sum_of_weights = 0.0;
    const double epsilon = 1e-6; // To avoid division by zero

    // Get the actual number of neighbors found (could be less than k if dataset is small)
    size_t neighbors_found = resultSet.size();

    if (neighbors_found == 0) {
        Warning("Nearest neighbor search found 0 neighbors.");
        return 0.0; // Default if no neighbors found
    }

    for (size_t i = 0; i < neighbors_found; ++i) {
        size_t neighbor_index = ret_indices[i];
        float distance_sq = out_dist_sqr[i];

        // Validate index
        if (neighbor_index >= point_cloud.target_currents.size()) {
             Warning("Nearest neighbor search returned invalid index: " + std::to_string(neighbor_index) +
                     ". Target Size: " + std::to_string(point_cloud.target_currents.size()) + ". Skipping this neighbor.");
             continue;
        }

        float neighbor_current = point_cloud.target_currents[neighbor_index];
        double weight = 1.0 / (distance_sq + epsilon);

        weighted_sum_currents += weight * neighbor_current;
        sum_of_weights += weight;

        // Debug: Print neighbor info
        // Debug("Neighbor " + std::to_string(i) + ": index=" + std::to_string(neighbor_index) +
        //       ", dist_sq=" + std::to_string(distance_sq) + ", current=" + std::to_string(neighbor_current) +
        //       ", weight=" + std::to_string(weight));
    }

    if (sum_of_weights == 0.0) {
         Warning("Sum of weights in NN weighted average is zero. This might happen if all distances are extremely large or weights are zero.");
         // Fallback: return the current of the closest neighbor found, if any
         if (neighbors_found > 0 && ret_indices[0] < point_cloud.target_currents.size()) {
             return point_cloud.target_currents[ret_indices[0]];
         } else {
             return 0.0; // Default fallback
         }
    }

    double predicted_current_normalised = weighted_sum_currents / sum_of_weights;

    // <<< EDIT 8: Denormalize the prediction >>>
    // Check if normalization was enabled (i.e., target mean/std were loaded)
    if (point_cloud.normalization_enabled) {
        // Denormalize: raw = normalized * std + mean
        double predicted_current_raw = predicted_current_normalised * point_cloud.target_std + point_cloud.target_mean;
         // Debug("Predicted Current (normalized): " + std::to_string(predicted_current_normalised));
         // Debug("Predicted Current (denormalized): " + std::to_string(predicted_current_raw));
        return predicted_current_raw;
    } else {
        Warning("Target normalization data was not loaded. Returning normalized prediction.");
        return predicted_current_normalised; // Return normalized if stats weren't loaded
    }

}

void TestMapping::LogDevianceData() {
    if (!log_deviance_data || !deviance_log_file.is_open()) {
        return; // Logging not enabled or file not open
    }

    // Helper lambda to serialize only the controlled servo elements of a matrix to a JSON array string
    auto serialize_controlled_servos = [this](const matrix& m) -> std::string {
        std::stringstream ss;
        ss << "[";
        bool first = true;
        for (int i = 0; i < current_controlled_servos.size(); ++i) {
            int servo_index = current_controlled_servos[i];
            if (servo_index < m.size()) {
                if (!first) ss << ", ";
                ss << std::fixed << std::setprecision(4) << m(servo_index);
                first = false;
            }
        }
        ss << "]";
        return ss.str();
    };

    auto serialise_imu_data = [this](const matrix& m) -> std::string {
        std::stringstream ss;
        ss << "[";
        bool first = true;
        for (int i = 0; i < m.size(); ++i) {
            if (!first) ss << ", ";
            ss << std::fixed << std::setprecision(4) << m(i);
            first = false;
        }
        ss << "]";
        return ss.str();
    };
    
    // Use a temporary stringstream to build the current JSON object
    std::stringstream current_object_ss;
    current_object_ss << "    {\n";
    current_object_ss << "      \"tick\": " << GetTick() << ",\n";
    current_object_ss << "      \"time\": " << std::fixed << std::setprecision(4) << GetNominalTime() << ",\n";
    current_object_ss << "      \"transition\": " << transition << ",\n";
    current_object_ss << "      \"present_position\": " << serialize_controlled_servos(present_position) << ",\n";
    current_object_ss << "      \"goal_position\": " << serialize_controlled_servos(goal_position_out) << ",\n";
    current_object_ss << "      \"present_current\": " << serialize_controlled_servos(present_current) << ",\n";
    current_object_ss << "      \"model_prediction\": " << serialize_controlled_servos(model_prediction) << ",\n"; // Last element in object
    current_object_ss << "      \"gyro\": " << serialise_imu_data(gyro) << ",\n";
    current_object_ss << "      \"accel\": " << serialise_imu_data(accel) << ",\n";
    current_object_ss << "      \"angles\": " << serialise_imu_data(eulerAngles) << ",\n";
    current_object_ss << "      \"starting_positions\": " << serialize_controlled_servos(starting_positions[transition]) << "\n";
    current_object_ss << "    }"; // Current object ends here, no "]}"

    // Add the current object to the deviance_log_stream buffer
    // If the buffer already contains objects, prefix the new one with a comma and newline
    if (deviance_log_stream.tellp() > 0) {
        deviance_log_stream << ",\n";
    }
    deviance_log_stream << current_object_ss.str();
    
    deviance_log_tick_counter++;
    if (deviance_log_tick_counter >= DEVIANCE_LOG_FLUSH_INTERVAL) {
        // Time to write the buffer to the file
        
        // Seek to before the closing "\\n  ]\\n}\\n" (7 characters)
        deviance_log_file.seekp(-7L, std::ios::end); 

        // If this is not the absolute first entry being written to the file, 
        // a comma is needed to separate it from previous (already flushed) entries.
        if (!is_first_deviance_log_entry) {
            deviance_log_file << ",\n"; 
        }
        
        deviance_log_file << deviance_log_stream.str(); // Write the buffered objects
        
        deviance_log_stream.str(""); // Clear the buffer
        deviance_log_stream.clear(); 
        deviance_log_tick_counter = 0; 
        
        is_first_deviance_log_entry = false; // Mark that at least one entry/batch has now been written to the file

        deviance_log_file << "\n  ]\n}\n"; // Re-add the closing structure to keep the JSON valid
        deviance_log_file.flush(); // Ensure it's written to disk
    }
}
// <<< END EDIT 6 >>>

INSTALL_CLASS(TestMapping)


