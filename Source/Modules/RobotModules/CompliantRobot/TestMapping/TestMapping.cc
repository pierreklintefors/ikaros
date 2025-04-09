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

// --- Nanoflann Point Cloud Adaptor ---
// Structure to hold the NN feature data and target currents
struct PointCloud
{
	// Feature points (inner vector has 13 dimensions)
	std::vector<std::vector<float>>  pts;
	// Corresponding target current values for each point
	std::vector<float> target_currents;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	// Since pts[idx] is std::vector<float>, we can just return pts[idx][dim]
	inline float kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		if (idx < pts.size() && dim < pts[idx].size()) {
			return pts[idx][dim];
		}
		// Return a default or handle error if index/dim is out of bounds
		// For simplicity, returning 0. Proper error handling might be needed.
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
#define FLOAT_COUNT 21  // 17 inputs + 4 predictions
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
    int position_margin = 3;
    int transition = 0;
    int starting_current = 30;
    int current_limit = 1700;
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
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud>, // Distance metric (L2 = Euclidean)
        PointCloud,                                    // Dataset adaptor
        NN_DIMS                                        // Dimensionality
        >;

    // Unique pointers to hold the built k-d tree indices
    std::unique_ptr<my_kd_tree_t> nn_tilt_index;
    std::unique_ptr<my_kd_tree_t> nn_pan_index;

    // We no longer need separate nn_tilt_data/nn_pan_data matrices or feature index vectors,
    // as this info is managed within PointCloud and the k-d tree structure.
    // int nn_tilt_current_col_idx = -1; // No longer needed directly
    // int nn_pan_current_col_idx = -1;  // No longer needed directly
    // std::vector<int> nn_tilt_feature_indices; // No longer needed
    // std::vector<int> nn_pan_feature_indices;  // No longer needed
     // --- End Nearest Neighbor data ---

    // <<< EDIT 1: Add members for deviance logging >>>
    std::ofstream deviance_log_file;       // File stream for logging
    std::stringstream deviance_log_stream; // Stringstream buffer for JSON content
     // Flag to enable logging logic
    int deviance_log_tick_counter = 0;     // Counter for periodic flushing
    const int DEVIANCE_LOG_FLUSH_INTERVAL = 100; // How often to flush (ticks)
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
        std::map<std::string, int> model_coefficients = {{"Linear", 5}, {"Quadratic", 7}};
        int num_coefficients = model_coefficients[model];

        // Initialize matrix with coefficients for each servo
        matrix coefficients_matrix(current_controlled_servos.size(), num_coefficients);
        
        // Iterate through each servo
        for (int i = 0; i < current_controlled_servos.size(); i++)
        {
            std::string servo_name = servo_names[current_controlled_servos(i)];

            

            // Get the coefficients dictionary for this servo
            dictionary servo_coeffs = coefficients[model][servo_name];

            // Skip "sigma" if it exists and iterate over the coefficient values
            int coeff_idx = 0;
            for ( auto &coeff : servo_coeffs)
            {
                if (coeff.first != "sigma")
                { // Skip sigma parameter
                    coefficients_matrix(i, coeff_idx) = (coeff.second).as_float();
                    if (i == 0)
                        coefficients_matrix.push_label(1, coeff.first);
                    coeff_idx++;
                }
            }
        }
        coefficients_matrix.set_name("CoefficientMatrix");
       

        return coefficients_matrix;
    }

    matrix SetGoalCurrent(matrix present_current, int increment, int limit, matrix position, matrix goal_position, int margin, matrix coefficients, std::string model_name, matrix model_prediction)
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

                // Get standardization parameters
                 // Ensure coefficient matrix is valid
                 if (coefficients.rows() <= i || coefficients.cols() < 7) {
                     Error("Coefficient matrix invalid for servo index " + std::to_string(i));
                     continue; // Skip this servo
                 }
                double current_mean = coefficients(i, 0); // Assuming column 0 is CurrentMean
                double current_std = coefficients(i, 1);  // Assuming column 1 is CurrentStd

                // Check for zero standard deviation
                if (current_std == 0) {
                     Warning("Current standard deviation is zero for servo index " + std::to_string(i) + ". Using mean as estimate.");
                     estimated_current = current_mean;
                 } else {
                     // Calculate standardized inputs
                    double distance_to_goal = goal_position(servo_idx) - position(servo_idx);
                     // Note: The standardization in the original code used current_mean/std for position and distance.
                     // This might be incorrect. Usually, you'd use position_mean/std and distance_mean/std.
                     // Replicating original logic here, but it might need review based on how coefficients were trained.
                    double std_position = (position(servo_idx) - current_mean) / current_std;
                    double std_distance = (distance_to_goal - current_mean) / current_std;


                    // Calculate linear terms (indices based on CreateCoefficientsMatrix)
                    estimated_current = coefficients(i, 6) + // intercept (index 6)
                                        coefficients(i, 2) * std_distance + // betas_linear[DistanceToGoal] (index 2)
                                        coefficients(i, 3) * std_position;  // betas_linear[Position] (index 3)


                    // Add quadratic terms if applicable
                    if (model_name == "Quadratic")
                    {
                        // betas_quad[DistanceToGoal_squared] (index 4), betas_quad[Position_squared] (index 5)
                        estimated_current += coefficients(i, 4) * std::pow(std_distance, 2) +
                                             coefficients(i, 5) * std::pow(std_position, 2);
                    }


                    // Unstandardize the result
                    estimated_current = estimated_current * current_std + current_mean;
                }

            }
            // <<< EDIT 7: Add NearestNeighbor logic >>>
            else if (model_name == "NearestNeighbor") {
                 // Check if indices were built successfully in Init
                 // Use servo_idx to determine which index/cloud to use
                 bool index_ready = (servo_idx == 0 && nn_tilt_index) || (servo_idx == 1 && nn_pan_index);

                 if (!index_ready) {
                     Error("NN k-d tree index not ready for servo " + std::to_string(servo_idx));
                     continue; // Skip NN estimation if index isn't built
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
                     // This case should be caught earlier or handled, but for safety:
                      Warning("NearestNeighbor model only implemented for NeckTilt (0) and NeckPan (1). Servo " + std::to_string(servo_idx) + " not handled.");
                      estimated_current = 0;
                      continue;
                 }

                // --- Construct the query vector as std::vector<float> ---
                 std::vector<float> query_point_vec(NN_DIMS); // Expected 17 features
                 int query_idx = 0;

                 // Generic sensor data (first 9 features)
                 if (gyro.size() < 3) { Error("Gyro data has insufficient size (<3) for NN query"); continue; }
                 for(int g=0; g<3; ++g) query_point_vec[query_idx++] = gyro(g);

                 if (accel.size() < 3) { Error("Accel data has insufficient size (<3) for NN query"); continue; }
                 for(int a=0; a<3; ++a) query_point_vec[query_idx++] = accel(a);

                 if (eulerAngles.size() < 3) { Error("EulerAngles data has insufficient size (<3) for NN query"); continue; }
                 for(int e=0; e<3; ++e) query_point_vec[query_idx++] = eulerAngles(e);

                 // For each servo in current_controlled_servos, add its features
                 for (int i = 0; i < current_controlled_servos.size(); ++i) {
                     int servo_idx = current_controlled_servos(i);
                     
                     // Add servo-specific features (4 per servo)
                     query_point_vec[query_idx++] = position(servo_idx); // Position
                     query_point_vec[query_idx++] = goal_position(servo_idx) - position(servo_idx); // DistToGoal
                     query_point_vec[query_idx++] = goal_position(servo_idx); // GoalPosition
                     
                     // Add starting position if available
                     if (transition < starting_positions.rows() && i < starting_positions.cols()) {
                         query_point_vec[query_idx++] = starting_positions(transition, i); // StartPosition
                     } else {
                         Error("Starting position data out of bounds for NN query (t=" + std::to_string(transition) + ", i=" + std::to_string(i) + ")");
                         query_point_vec[query_idx++] = 0.0f;
                     }
                 }
                 // --- End query vector construction ---

                 if (query_idx != NN_DIMS) {
                     Error("Internal error: Mismatch building NN query vector size (" + std::to_string(query_idx) +
                           ") vs expected features (" + std::to_string(NN_DIMS) + ") for servo " + std::to_string(servo_idx));
                     continue;
                 }

                 // Perform the nearest neighbor search using the k-d tree
                 estimated_current = FindNearestNeighborCurrent(query_point_vec, *current_index_ptr, *current_point_cloud_ptr);

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
             model_prediction(servo_idx) = std::abs(estimated_current);
        }

        return current_output;
    }

    void StartPythonProcess()
    {
        std::string directory = __FILE__;
        // Get directory path just like you already do
        directory = directory.substr(0, directory.find_last_of("/"));
        std::string envPath = directory + "/tensorflow_venv/bin/python3";
        std::string pythonPath = directory + "/ANN_prediction.py";

        child_pid = fork();
        if (child_pid < 0) {
             Error("Failed to fork process for Python script");
             return;
        }
        if (child_pid == 0) {
             // Child process: replace the current process image with the Python script.
             execl(envPath.c_str(), "python", pythonPath.c_str(), (char *)nullptr);
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
        
        //go up in the directory to get to the folder containing the coefficients.json file
        std::string coefficientsPath = scriptPath.substr(0, scriptPath.find_last_of("/"));
        coefficientsPath = coefficientsPath.substr(0, coefficientsPath.find_last_of("/"));
        coefficientsPath = coefficientsPath + "/CurrentPositionMapping/models/coefficients.json";
        current_coefficients.load_json(coefficientsPath);

        

        number_transitions = num_transitions.as_int()+1; // Add one to the number of transitions to include the ending position
        position_transitions.set_name("PositionTransitions");
        position_transitions = RandomisePositions(number_transitions, min_limits, max_limits, robotType.as_string());
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
        coeffcient_matrix = CreateCoefficientsMatrix(current_coefficients, current_controlled_servos, prediction_model.as_string(), servo_names);
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

            // Print debug information
            Debug("Shared memory initialized:\n");
            Debug("- Required size: " + std::to_string(required_size) + " bytes\n");
            Debug("- SharedData size: " + std::to_string(sizeof(SharedData)) + " bytes\n");

            // Start Python process
            StartPythonProcess();
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
            std::string tilt_data_path = dataBasePath + "tilt_filtered_data_raw.csv";
            Debug("Loading Tilt NN data from: " + tilt_data_path);
            std::vector<std::string> tilt_feature_labels = {
                "GyroX", "GyroY", "GyroZ", "AccelX", "AccelY", "AccelZ",
                "AngleX", "AngleY", "AngleZ", 
                "TiltPosition", "TiltDistToGoal", "TiltGoalPosition", "TiltStartPosition",
                "PanPosition", "PanDistToGoal", "PanGoalPosition", "PanStartPosition"  // Added pan features
            };
            std::string tilt_target_label = "TiltCurrent";

            tilt_load_success = LoadCSVData(tilt_data_path, tilt_feature_labels, tilt_target_label, nn_tilt_point_cloud);

            if(tilt_load_success) {
                 Debug("Building k-d tree index for Tilt data (" + std::to_string(nn_tilt_point_cloud.pts.size()) + " points)...");
                 // Construct the k-d tree index
                 nn_tilt_index = std::make_unique<my_kd_tree_t>(NN_DIMS, nn_tilt_point_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max_leaf */) );
                 nn_tilt_index->buildIndex();
                 Debug("Tilt k-d tree built.");
            } else {
                Error("Failed to load Tilt NN data. Nearest neighbor search for tilt will not work.");
                // Decide if we should return or continue without tilt NN
                // return; // Option: Stop initialization if tilt data fails
            }


             // --- Load Data and Build Index for Pan ---
             std::string pan_data_path = dataBasePath + "pan_filtered_data_raw.csv";
             Debug("Loading Pan NN data from: " + pan_data_path);
              std::vector<std::string> pan_feature_labels = {
                  "GyroX", "GyroY", "GyroZ", "AccelX", "AccelY", "AccelZ",
                  "AngleX", "AngleY", "AngleZ",
                  "TiltPosition", "TiltDistToGoal", "TiltGoalPosition", "TiltStartPosition",  // Added tilt features
                  "PanPosition", "PanDistToGoal", "PanGoalPosition", "PanStartPosition"
             };
              std::string pan_target_label = "PanCurrent";

              pan_load_success = LoadCSVData(pan_data_path, pan_feature_labels, pan_target_label, nn_pan_point_cloud);

             if (pan_load_success) {
                 Debug("Building k-d tree index for Pan data (" + std::to_string(nn_pan_point_cloud.pts.size()) + " points)...");
                 // Construct the k-d tree index
                 nn_pan_index = std::make_unique<my_kd_tree_t>(NN_DIMS, nn_pan_point_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max_leaf */) );
                 nn_pan_index->buildIndex();
                 Debug("Pan k-d tree built.");
             } else {
                 Error("Failed to load Pan NN data. Nearest neighbor search for pan will not work.");
                 // return; // Option: Stop initialization if pan data fails
             }
             // Check if both failed?
             if (!tilt_load_success && !pan_load_success) {
                 Error("Failed to load data for both Tilt and Pan. NN model unusable.");
                 return; // Definitely stop if both failed
             }

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
                     ss_time << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
                     std::string time_stamp_str = ss_time.str();

                    std::string filepath = resultsDir + "/deviance_log_" + time_stamp_str + "_" + std::to_string(unique_id) + ".json";

                    deviance_log_file.open(filepath, std::ios::out | std::ios::trunc);
                    if (!deviance_log_file.is_open()) {
                        Error("Failed to open deviance log file: " + filepath);
                        log_deviance_data = false; // Disable logging on failure
                    } else {
                        Debug("Opened deviance log file: " + filepath);
                        // Start the JSON structure
                        deviance_log_stream << "{\n  \"deviance_data\": [\n";
                        is_first_deviance_log_entry = true; // Reset comma flag
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

            transition++;
            
            // Save data after every 5th transition
            if (transition % 5 == 0) {
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

                goal_position_out.copy(position_transitions[transition]);
                goal_current.copy(present_current);

                // Compute and store the predicted goal current from the model for this new transition
                // Only do this once per transition
                matrix all_predicted = SetGoalCurrent(present_current, current_increment, current_limit,
                                                      present_position, goal_position_out, position_margin,
                                                      coeffcient_matrix, std::string(prediction_model), model_prediction);
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
                
                // Distance to goal
                shared_data->data[idx++] = abs((float)goal_position_out[servo_idx] - (float)present_position[servo_idx]);
                
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
            int timeout_ms = 5;  // 5ms timeout
            auto start = std::chrono::steady_clock::now();
            bool got_response = false;
            
            while (!got_response) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
                
                if (elapsed > timeout_ms) {
                    Debug("Timeout waiting for ANN prediction. Elapsed time: " + std::to_string(elapsed) + "ms");
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
                        int regular_offset = prediction_base_offset + (i * 2);
                        int start_offset = regular_offset + 1;
                        
                        // Update predictions
                        model_prediction[servo_idx] = abs(shared_data->data[regular_offset]);
                        model_prediction_start[servo_idx] = abs(shared_data->data[start_offset]);
                        
                        // Build debug message
                        if (i > 0) debug_msg += ", ";
                        debug_msg += std::string(servo_names[servo_idx]) + 
                                    "=(regular=" + std::to_string(model_prediction[servo_idx]) + 
                                    ", start=" + std::to_string(model_prediction_start[servo_idx]) + ")";
                    }
                    
                    got_response = true;
                    Debug(debug_msg);
                }

                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }

            if (!got_response) {
                Debug("Using previous ANN predictions due to timeout");
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
                        
                        if (abs(current_pos - start_pos) <= position_margin) {
                            // Use starting current from model_prediction_start matrix
                            goal_current(servo_idx) = std::min<float>(model_prediction_start[servo_idx], (float)current_limit);
                            
                            // Add debug to verify the correct current is being used
                            Debug("Using STARTING current (close to start pos) for " + std::string(servo_names[servo_idx]) + 
                                  ": " + std::to_string(model_prediction_start[servo_idx]));
                        } else {
                            // Use regular current from model_prediction matrix
                            goal_current(servo_idx) = std::min<float>(model_prediction[servo_idx], (float)current_limit);
                            
                            // Add debug to verify the correct current is being used
                            Debug("Using REGULAR current (moved from start pos) for " + std::string(servo_names[servo_idx]) + 
                                  ": " + std::to_string(model_prediction[servo_idx]));
                        }
                    } else {
                        // For other models, use the existing prediction method
                        goal_current.copy(SetGoalCurrent(present_current, current_increment, current_limit,
                                                   present_position, goal_position_out, position_margin,
                                                   coeffcient_matrix, std::string(prediction_model), model_prediction));
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

            model_prediction.print();
            model_prediction_start.print();
            goal_current.print();
        }

        if (GetTick() >= 2 && predicted_goal_current.sum() == 0) {
            matrix all_predicted = SetGoalCurrent(present_current, current_increment, current_limit,
                                                  present_position, goal_position_out, position_margin,
                                                  coeffcient_matrix, std::string(prediction_model), model_prediction);
            
            for (int i = 0; i < current_controlled_servos.size(); i++) {
                int servo_idx = current_controlled_servos(i);
                predicted_goal_current(transition, i) = all_predicted(servo_idx);
            }
        }

        // <<< EDIT 4: Call LogDevianceData if enabled >>>
        // Log data *before* potentially changing state in this tick (like transition increment)
        if (log_deviance_data && GetTick() > 2) { // Start logging after initial ticks
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

        // <<< EDIT 5: Finalize and close deviance log file >>>
        if (log_deviance_data && deviance_log_file.is_open()) {
            // Write any remaining data from the buffer
            if (deviance_log_tick_counter > 0 || !deviance_log_stream.str().empty()) {
                 deviance_log_file << deviance_log_stream.str();
                 deviance_log_stream.str(""); // Clear buffer after writing
                 deviance_log_stream.clear();
            }
            // Close the JSON array and root object
            deviance_log_stream << "\n  ]\n}\n";
            deviance_log_file << deviance_log_stream.str(); // Write final closing brackets

            deviance_log_file.close();
            Debug("Closed deviance log file.");
        }
        // <<< END EDIT 5 >>>
    }

    // <<< EDIT 6: Add function signature >>>
    void LogDevianceData();
    // <<< END EDIT 6 >>>

    // <<< EDIT 3: Implement helper functions >>>
    // Helper function to load CSV data and populate the PointCloud structure
    // Returns true on success, false on failure.
    bool LoadCSVData(const std::string& filepath, const std::vector<std::string>& feature_labels, const std::string& target_label, PointCloud& point_cloud_out);
    double FindNearestNeighborCurrent(const std::vector<float>& query_point, const std::unique_ptr<my_kd_tree_t>& index_ptr, const PointCloud& point_cloud);
    // <<< END EDIT 3 >>>
};

// <<< EDIT 3: Implement helper functions >>>
// Helper function to load CSV data and populate the PointCloud structure
// Returns true on success, false on failure.
bool TestMapping::LoadCSVData(const std::string& filepath, const std::vector<std::string>& feature_labels, const std::string& target_label, PointCloud& point_cloud_out) {
    point_cloud_out.pts.clear();
    point_cloud_out.target_currents.clear();

    std::ifstream file(filepath);
    if (!file.is_open()) {
        Error("Failed to open CSV file: " + filepath);
        return false;
    }

    std::string line;
    std::vector<std::string> headers;
    int target_col_index = -1;
    std::vector<int> feature_indices(feature_labels.size(), -1);

    // Read header line
    if (std::getline(file, line)) {
        // Remove potential UTF-8 BOM if present
        if (line.size() >= 3 && line[0] == (char)0xEF && line[1] == (char)0xBB && line[2] == (char)0xBF) {
            line = line.substr(3);
        }
        // Remove trailing carriage return if present (Windows line endings)
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

            // Check if this header is our target
            if (header == target_label) {
                target_col_index = current_col;
            }
            // Check if this header is one of our features
            for (size_t j = 0; j < feature_labels.size(); ++j) {
                if (header == feature_labels[j]) {
                    if (feature_indices[j] != -1) {
                         Warning("Duplicate feature label '" + feature_labels[j] + "' found in CSV header: " + filepath);
                    }
                    feature_indices[j] = current_col;
                }
            }
            current_col++;
        }
    } else {
        Error("CSV file is empty or could not read header: " + filepath);
        return false;
    }

    // Validate that all required columns were found
    if (target_col_index == -1) {
        Error("Target label '" + target_label + "' not found in CSV header: " + filepath);
        return false;
    }
    std::string missing_features;
    for (size_t j = 0; j < feature_labels.size(); ++j) {
        if (feature_indices[j] == -1) {
            if (!missing_features.empty()) missing_features += ", ";
            missing_features += "'" + feature_labels[j] + "'";
        }
    }
     if (!missing_features.empty()) {
         Error("Feature label(s) " + missing_features + " not found in CSV header: " + filepath);
         return false;
     }
     if (feature_labels.size() != NN_DIMS) {
         Error("Number of feature labels (" + std::to_string(feature_labels.size()) + ") does not match expected NN_DIMS (" + std::to_string(NN_DIMS) + ")");
         return false;
     }


    // Read data rows
    while (std::getline(file, line)) {
        // Remove trailing carriage return if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue; // Skip empty lines

        std::stringstream ss(line);
        std::string cell;
        std::vector<float> all_row_values;
        while (std::getline(ss, cell, ',')) {
            try {
                 // Handle potential empty cells or non-numeric gracefully? For now, error out.
                 if (cell.empty()) {
                      Error("Empty cell encountered in CSV file: " + filepath);
                      return false;
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

        if (all_row_values.size() != headers.size()) {
            Warning("Row has different number of columns (" + std::to_string(all_row_values.size()) +
                    ") than header (" + std::to_string(headers.size()) + ") in CSV: " + filepath + ". Skipping row.");
            continue; // Skip inconsistent rows
        }

        // Extract feature values in the specified order and the target value
        std::vector<float> feature_values(NN_DIMS);
        bool row_ok = true;
        for(size_t j=0; j < NN_DIMS; ++j) {
            int col_idx = feature_indices[j];
             // This check should be redundant due to header validation, but good for safety
             if (col_idx < 0 || col_idx >= all_row_values.size()) {
                 Error("Internal error: Invalid feature index " + std::to_string(col_idx) + " while reading row.");
                 row_ok = false;
                 break;
             }
            feature_values[j] = all_row_values[col_idx];
        }
         // Check target index validity
         if (target_col_index < 0 || target_col_index >= all_row_values.size()) {
              Error("Internal error: Invalid target index " + std::to_string(target_col_index) + " while reading row.");
              row_ok = false;
         }

        if (row_ok) {
            point_cloud_out.pts.push_back(feature_values);
            point_cloud_out.target_currents.push_back(all_row_values[target_col_index]);
        }
    }

    file.close();

    if (point_cloud_out.pts.empty()) {
        Warning("No valid data rows loaded into PointCloud from CSV: " + filepath);
        return false; // Treat as failure if no points loaded
    }

    Debug("Loaded " + std::to_string(point_cloud_out.pts.size()) + " data points into PointCloud from " + filepath);

    // // Optional Debug: Print loaded feature indices
    // std::string feature_idx_str = "Feature column indices used: ";
    // for(int idx : feature_indices) feature_idx_str += std::to_string(idx) + " ";
    // Debug(feature_idx_str);
    // Debug("Target column index used: " + std::to_string(target_col_index));

    return true;
}

// Helper function to find the nearest neighbor using the k-d tree index
double TestMapping::FindNearestNeighborCurrent(
    const std::vector<float>& query_point, // Use std::vector for query
    const std::unique_ptr<my_kd_tree_t>& index_ptr, // Pass pointer to the index
    const PointCloud& point_cloud) // Pass associated point cloud for target lookup
{
    if (!index_ptr || query_point.size() != NN_DIMS) {
        Error("Invalid k-d tree index or query point dimension for nearest neighbor search.");
        return 0.0; // Return a default value or handle error
    }
     if (point_cloud.pts.empty()) {
          Warning("Point cloud associated with k-d tree index is empty.");
          return 0.0;
     }


    // Perform k-NN search: find 1 nearest neighbor
    size_t num_results = 1;
    size_t ret_index; // Index of the nearest neighbor in the PointCloud
    float out_dist_sqr; // Squared distance to the nearest neighbor

    // Prepare result set
    nanoflann::KNNResultSet<float> resultSet(num_results);
    resultSet.init(&ret_index, &out_dist_sqr);

    // Perform the search
    // query_point.data() returns a float* needed by findNeighbors
    // Use default search parameters by default-constructing SearchParams
    index_ptr->findNeighbors(resultSet, query_point.data(), nanoflann::SearchParameters()); // Use default SearchParams


    // Check if a neighbor was found and the index is valid
    // resultSet.size() might be 0 if no neighbor is found (e.g., empty dataset - though checked earlier)
    if (resultSet.size() > 0 && ret_index < point_cloud.target_currents.size()) {
        // Return the target current value from the best matching point
        return point_cloud.target_currents[ret_index];
    } else {
         // This case should ideally not happen if the index was built correctly and the dataset isn't empty.
         Warning("Nearest neighbor search failed or returned invalid index. Ret Index: " + std::to_string(ret_index) + ", Target Size: " + std::to_string(point_cloud.target_currents.size()));
        return 0.0; // Default if no neighbor found or index invalid
    }
}
// <<< END EDIT 3 >>>

// <<< EDIT 6: Add function implementation >>>
void TestMapping::LogDevianceData() {
    if (!log_deviance_data || !deviance_log_file.is_open()) {
        return; // Logging not enabled or file not open
    }

    // Add comma before adding a new object, except for the first one
    if (!is_first_deviance_log_entry) {
        deviance_log_stream << ",\n";
    } else {
        is_first_deviance_log_entry = false;
    }

    // Use stringstream for efficient JSON object creation
    deviance_log_stream << "    {\n";
    deviance_log_stream << "      \"tick\": " << GetTick() << ",\n";
    deviance_log_stream << "      \"time\": " << std::fixed << std::setprecision(4) << GetNominalTime() << ",\n"; // Use nominal time for consistency
    deviance_log_stream << "      \"transition\": " << transition << ",\n";

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

    deviance_log_stream << "      \"present_position\": " << serialize_controlled_servos(present_position) << ",\n";
    deviance_log_stream << "      \"goal_position\": " << serialize_controlled_servos(goal_position_out) << ",\n";
    deviance_log_stream << "      \"present_current\": " << serialize_controlled_servos(present_current) << ",\n";
    deviance_log_stream << "      \"model_prediction\": " << serialize_controlled_servos(model_prediction) << "\n"; // Last element, no comma
    deviance_log_stream << "    }"; // Close the JSON object

    // Increment counter and check if it's time to flush
    deviance_log_tick_counter++;
    if (deviance_log_tick_counter >= DEVIANCE_LOG_FLUSH_INTERVAL) {
        deviance_log_file << deviance_log_stream.str(); // Write buffer to file
        deviance_log_stream.str(""); // Clear the stringstream buffer
        deviance_log_stream.clear(); // Clear potential error states
        deviance_log_tick_counter = 0; // Reset counter
         // Optional: Flush the OS buffer to disk immediately (can impact performance)
         // deviance_log_file.flush();
    }
}
// <<< END EDIT 6 >>>

INSTALL_CLASS(TestMapping)


