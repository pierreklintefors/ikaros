#include "ikaros.h"
#include <random> // Include the header file that defines the 'Random' function
#include <fstream>
#include <vector>
#include <memory>
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


using namespace ikaros;

int TIMEOUT_DURATION = 5.0; // seconds



class TestMapping: public Module
{
 

    // Inputs
    matrix present_position;
    matrix present_current;
    matrix gyro;
    matrix accel;
    matrix eulerAngles;
    matrix ANN_prediction;
    matrix goal_position_in;
    matrix start_position_in;
    

    // Outputs
    matrix goal_current;
    matrix starting_positions;
    matrix model_prediction;

    // Internal
    matrix servos_to_test;
    matrix predicted_current_history;
    

    std::vector<std::shared_ptr<std::vector<float>>> moving_trajectory;
    matrix approximating_goal;
    matrix reached_goal;

    
   
    // Parameters
    parameter num_transitions_param;
    parameter robotType;
    parameter prediction_model;

    parameter static_test_mode;
    parameter static_countdown;
    parameter static_data_collection;
    parameter static_interval;
    
    // Internal
    std::random_device rd;

    int position_margin = 2;
    int transition = 0;
    int number_transitions;
  
    int unique_id;
    bool second_tick = false;
    bool first_start_position = true;
    std::string time_stamp_str_no_dots;
    int num_input_data = 0;
    matrix transition_start_time;
    matrix transition_duration;
    double time_prev_position;
    bool goal_changed = false;
    
   

    std::vector<std::string> servo_names;

    matrix initial_currents;
    matrix current_differences;

 
  
    matrix timeout_occurred;

    
   
    matrix prediction_error;

    // Static test mode variables
    enum StaticTestPhase { HOLDING, READY_TO_PUSH, PUSHING, RESTING };
    StaticTestPhase static_phase = HOLDING;
    double static_phase_start_time = 0.0;
    int static_test_cycle = 0;
    std::ofstream static_push_log_file;
    std::stringstream static_push_log_stream;
    bool is_first_static_log_entry = true;
    matrix static_initial_position;  // Store initial position to maintain throughout test

 
 

    std::ofstream deviance_log_file;       // File stream for logging
    std::stringstream deviance_log_stream; // Stringstream buffer for JSON content
     // Flag to enable logging logic
    int deviance_log_tick_counter = 0;     // Counter for periodic flushing
    const int DEVIANCE_LOG_FLUSH_INTERVAL = 2; // How often to flush (ticks)
    bool is_first_deviance_log_entry = true; // To handle commas in JSON array

    bool GoalHasChanged(matrix goal_position)
    {
        // On first tick, there's no previous goal - return false
        if (GetTick() <= 1) {
            return false;
        }
        return (goal_position != goal_position.last());
    }

    void ReachedGoal(matrix present_position, matrix goal_positions, matrix &reached_goal, int margin){
        // Add safety check
        if (present_position.size() != goal_positions.size() || 
            present_position.size() <= 0 || 
            reached_goal.size() <= 0) {
            Error("Invalid matrix dimensions in ReachedGoal");
            return;
        }
        
        for (int i = 0; i < servos_to_test.size(); i++){
            int servo_idx = servos_to_test(i);
            // Add bounds check
            if (servo_idx < 0 || servo_idx >= present_position.size() || 
                servo_idx >= goal_positions.size()) {
                Error("Servo index out of bounds: " + std::to_string(servo_idx));
                continue;
            }
            
            if (reached_goal(servos_to_test(i)) == 0 &&
                abs(present_position(servos_to_test(i)) - goal_positions(servos_to_test(i))) < margin){

                reached_goal(servos_to_test(i)) = 1;

                double current_time_ms = GetTime();
                
                if (transition < transition_duration.rows() && i < transition_duration.cols()) {
                    transition_duration(transition, i) = current_time_ms - transition_start_time(transition, i);
                }
            }
        }
    }

    matrix ApproximatingGoal(matrix &present_position,  matrix goal_position, int margin){
        // On first tick, there's no previous position - just return current approximating_goal
        if (GetTick() <= 1) {
            return approximating_goal;
        }
        
        matrix previous_position = present_position.last();
        for (int i =0; i < servos_to_test.size(); i++){
            int servo_idx = servos_to_test(i);
            
            // Add bounds checking
            if (servo_idx < 0 || servo_idx >= goal_position.size() || 
                servo_idx >= present_position.size() || 
                servo_idx >= previous_position.size() ||
                i >= approximating_goal.size()) {
                continue;
            }
            
            //Checking if distance to goal is decreasing
            if (abs(goal_position(servo_idx) - present_position(servo_idx)) < 
                0.97*abs(goal_position(servo_idx) - previous_position(servo_idx))){
                Debug( "Approximating Goal");
                approximating_goal(servos_to_test(i)) = 1;
                
            }
            else{
                approximating_goal(servos_to_test(i)) = 0;
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

    void LogDevianceData()
    {
        if (!deviance_log_file.is_open())
        {
            return; // Logging not enabled or file not open
        }

        // Helper lambda to serialize only the controlled servo elements of a matrix to a JSON array string
        auto serialize_controlled_servos = [this](const matrix &m) -> std::string
        {
            std::stringstream ss;
            ss << "[";
            bool first = true;
            for (int i = 0; i < servos_to_test.size(); ++i)
            {
                int servo_index = servos_to_test[i];
                if (servo_index < m.size())
                {
                    if (!first)
                        ss << ", ";
                    ss << std::fixed << std::setprecision(4) << m(servo_index);
                    first = false;
                }
            }
            ss << "]";
            return ss.str();
        };

        auto serialise_imu_data = [this](const matrix &m) -> std::string
        {
            std::stringstream ss;
            ss << "[";
            bool first = true;
            for (int i = 0; i < m.size(); ++i)
            {
                if (!first)
                    ss << ", ";
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
        current_object_ss << "      \"time\": " << std::fixed << std::setprecision(4) << GetTime() << ",\n";
        current_object_ss << "      \"transition\": " << transition << ",\n";
        current_object_ss << "      \"present_position\": " << serialize_controlled_servos(present_position) << ",\n";
        current_object_ss << "      \"goal_position\": " << serialize_controlled_servos(goal_position_in) << ",\n";
        current_object_ss << "      \"present_current\": " << serialize_controlled_servos(present_current) << ",\n";
        current_object_ss << "      \"model_prediction\": " << serialize_controlled_servos(model_prediction) << ",\n"; // Last element in object
        current_object_ss << "      \"gyro\": " << serialise_imu_data(gyro) << ",\n";
        current_object_ss << "      \"accel\": " << serialise_imu_data(accel) << ",\n";
        current_object_ss << "      \"angles\": " << serialise_imu_data(eulerAngles) << ",\n";
        current_object_ss << "      \"starting_positions\": " << serialize_controlled_servos(starting_positions[transition]) << "\n";
        current_object_ss << "    }"; // Current object ends here, no "]}"

        // Add the current object to the deviance_log_stream buffer
        // If the buffer already contains objects, prefix the new one with a comma and newline
        if (deviance_log_stream.tellp() > 0)
        {
            deviance_log_stream << ",\n";
        }
        deviance_log_stream << current_object_ss.str();

        deviance_log_tick_counter++;
        if (deviance_log_tick_counter >= DEVIANCE_LOG_FLUSH_INTERVAL)
        {
            // Time to write the buffer to the file

            // Seek to before the closing "\\n  ]\\n}\\n" (7 characters)
            deviance_log_file.seekp(-7L, std::ios::end);

            // If this is not the absolute first entry being written to the file,
            // a comma is needed to separate it from previous (already flushed) entries.
            if (!is_first_deviance_log_entry)
            {
                deviance_log_file << ",\n";
            }

            deviance_log_file << deviance_log_stream.str(); // Write the buffered objects

            deviance_log_stream.str(""); // Clear the buffer
            deviance_log_stream.clear();
            deviance_log_tick_counter = 0;

            is_first_deviance_log_entry = false; // Mark that at least one entry/batch has now been written to the file

            deviance_log_file << "\n  ]\n}\n"; // Re-add the closing structure to keep the JSON valid
            deviance_log_file.flush();         // Ensure it's written to disk
        }
    }
    // <<< END EDIT 6 >>>

    void LogStaticPushData()
    {
        if (!static_test_mode || !static_push_log_file.is_open())
        {
            return;
        }

        // Helper lambda to serialize controlled servo elements
        auto serialize_controlled_servos = [this](const matrix &m) -> std::string
        {
            std::stringstream ss;
            ss << "[";
            bool first = true;
            for (int i = 0; i < servos_to_test.size(); ++i)
            {
                int servo_index = servos_to_test[i];
                if (servo_index < m.size())
                {
                    if (!first)
                        ss << ", ";
                    ss << std::fixed << std::setprecision(4) << m(servo_index);
                    first = false;
                }
            }
            ss << "]";
            return ss.str();
        };

        auto serialize_imu_data = [this](const matrix &m) -> std::string
        {
            std::stringstream ss;
            ss << "[";
            bool first = true;
            for (int i = 0; i < m.size(); ++i)
            {
                if (!first)
                    ss << ", ";
                ss << std::fixed << std::setprecision(4) << m(i);
                first = false;
            }
            ss << "]";
            return ss.str();
        };

        // Build JSON object for this data point
        std::stringstream entry_ss;

        // Add comma if not first entry
        if (!is_first_static_log_entry)
        {
            entry_ss << ",\n";
        }

        entry_ss << "    {\n";
        entry_ss << "      \"cycle\": " << static_test_cycle << ",\n";
        entry_ss << "      \"tick\": " << GetTick() << ",\n";
        entry_ss << "      \"time\": " << std::fixed << std::setprecision(4) << GetTime() << ",\n";
        entry_ss << "      \"phase_elapsed\": " << std::fixed << std::setprecision(4)
                 << (GetTime() - static_phase_start_time) << ",\n";
        entry_ss << "      \"present_position\": " << serialize_controlled_servos(present_position) << ",\n";
        entry_ss << "      \"goal_position\": " << serialize_controlled_servos(goal_position_in) << ",\n";
        entry_ss << "      \"present_current\": " << serialize_controlled_servos(present_current) << ",\n";

        // Add ANN prediction if available
        if (ANN_prediction.connected() && ANN_prediction.size() > 0)
        {
            entry_ss << "      \"ann_prediction\": " << serialize_controlled_servos(ANN_prediction) << ",\n";
        }

        // Add model prediction if available
        if (model_prediction.size() > 0)
        {
            entry_ss << "      \"model_prediction\": " << serialize_controlled_servos(model_prediction) << ",\n";
        }

        entry_ss << "      \"gyro\": " << serialize_imu_data(gyro) << ",\n";
        entry_ss << "      \"accel\": " << serialize_imu_data(accel) << ",\n";
        entry_ss << "      \"euler_angles\": " << serialize_imu_data(eulerAngles) << "\n";
        entry_ss << "    }";

        // Write to file
        static_push_log_file << entry_ss.str();
        static_push_log_file.flush();

        is_first_static_log_entry = false;
    }

    void Init()
    {
        //IO
        Bind(present_current, "PresentCurrent");
        Bind(present_position, "PresentPosition");
        Bind(gyro, "GyroData");
        Bind(accel, "AccelData");
        Bind(eulerAngles, "EulerAngles");
        Bind(goal_position_in, "GoalPositionIn");
        Bind(start_position_in, "StartPositionIn");
        Bind(model_prediction, "ModelPrediction");
    
        Bind(ANN_prediction, "ANN_prediction");


        //parameters
        Bind(num_transitions_param, "NumberTransitions");
        number_transitions = num_transitions_param.as_int();
        Bind(robotType, "RobotType");
        Bind(prediction_model, "CurrentPrediction");

        Bind(static_test_mode, "StaticTestMode");
        Bind(static_countdown, "StaticCountdown");
        Bind(static_data_collection, "StaticDataCollection");
        Bind(static_interval, "StaticInterval");


        std::string scriptPath = __FILE__;
        
       
      

        if (present_position.size() == 0)
        {
            Error("present_position is empty. Ensure that the input is connected and has valid dimensions.");
            return;
        }

        servo_names = {"NeckTilt", "NeckPan", "LeftEye", "RightEye", "LeftPupil", "RightPupil", "LeftArmJoint1", "LeftArmJoint2", "LeftArmJoint3", "LeftArmJoint4", "LeftArmJoint5", "LeftHand", "RightArmJoint1", "RightArmJoint2", "RightArmJoint3", "RightArmJoint4", "RightArmJoint5", "RightHand", "Body"};
        

        if(robotType.as_string() == "Torso"){
            servos_to_test.set_name("TestedServosTorso");
            servos_to_test = {0, 1};
            
        }
        else if(robotType.as_string() == "FullBody"){
            servos_to_test.set_name("TestServosFullBody");
            servos_to_test = {0, 1, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18};
        }
        else{
            Error("Robot type not recognized");
        }
        
        // Initialize matrices AFTER servos_to_test is set
        // Size these to present_position.size() since they're indexed by servo_idx (0-18), not loop index
        reached_goal = matrix(present_position.size());
        reached_goal.set(0);
        reached_goal.set_name("ReachedGoal");
        
        approximating_goal = matrix(present_position.size());
        approximating_goal.set(0);
        approximating_goal.set_name("ApproximatingGoal");
        
        transition_start_time = matrix(number_transitions, servos_to_test.size());
        transition_start_time.set(0);
        transition_start_time.set_name("StartedTransitionTime");
        
        transition_duration = matrix(number_transitions, servos_to_test.size());
        transition_duration.set(0);
        transition_duration.set_name("TransitionDuration");

        unique_id = GenerateRandomNumber(0, 1000000);

    

        // Initialize new matrices for tracking prediction data
        initial_currents = matrix(number_transitions, servos_to_test.size());
        current_differences = matrix(number_transitions, servos_to_test.size());
        initial_currents.set(0);
        current_differences.set(0);

        // New: Initialize matrices to track the model's predicted values and prediction errors
        predicted_current_history.set_name("PredictedGoalCurrent");
        predicted_current_history = matrix(number_transitions, servos_to_test.size());
        predicted_current_history.set(0);

        prediction_error.set_name("PredictionError");
        prediction_error = matrix(number_transitions, servos_to_test.size());
        prediction_error.set(0);

        model_prediction.set_name("ModelPrediction");
        // Initialize model_prediction with the same size as present_current, then set to 0
        
        
        if (std::string(prediction_model) == "ANN") {
            if (ANN_prediction.connected()) {
                // Check if present_current has valid size
                if (present_current.size() == 0) {
                    Error("present_current has no data during Init - cannot initialize model_prediction");
                    return;
                }
                // Initialize model_prediction to zeros
                model_prediction.copy(present_current);
                model_prediction.set(0);
                
                // Map ANN predictions to the controlled servo positions
                // ANN_prediction should have predictions for each controlled servo
                if (ANN_prediction.size() >= servos_to_test.size()) {
                    for (int i = 0; i < servos_to_test.size(); i++) {
                        int servo_idx = servos_to_test(i);
                        model_prediction(servo_idx) = ANN_prediction(i);
                    }
                } else {
                    Error("ANN_prediction size (" + std::to_string(ANN_prediction.size()) + 
                          ") is less than controlled servos (" + std::to_string(servos_to_test.size()) + ")");
                    return;
                }
            }
            else {
                Error("ANN_prediction is not connected");
                return;
            }
        }
      
        timeout_occurred.set_name("TimeoutOccurred");
        timeout_occurred = matrix(number_transitions, servos_to_test.size());
        timeout_occurred.set(0);

        // Initialize transition_start_time as a 2D matrix
        transition_start_time.set_name("StartedTransitionTime");
        transition_start_time = matrix(number_transitions, servos_to_test.size());
        transition_start_time.set(0);

        starting_positions = matrix( number_transitions, servos_to_test.size());
        starting_positions.set(0);

        
        
        // Initialize static test mode if enabled
        if (static_test_mode) {
            try {
                std::string scriptPath = __FILE__;
                std::string scriptDirectory = scriptPath.substr(0, scriptPath.find_last_of("/\\"));
                std::string resultsDir = scriptDirectory + "/results/static_push_logs";

                // Create directory if it doesn't exist
                if (!std::filesystem::exists(resultsDir)) {
                    if (!std::filesystem::create_directories(resultsDir)) {
                         Error("Failed to create static push results directory: " + resultsDir);
                         static_test_mode = false;
                    }
                }

                if (static_test_mode) {
                    // Get current time for unique filename
                     auto now = std::chrono::system_clock::now();
                     auto now_c = std::chrono::system_clock::to_time_t(now);
                     std::stringstream ss_time;
                     ss_time << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
                     std::string time_stamp_str = ss_time.str();

                    std::string filepath = resultsDir + "/static_push_log_" + time_stamp_str + ".json";
                    static_initial_position = matrix(present_position.size());
                    static_initial_position.set(0);
                    static_push_log_file.open(filepath, std::ios::out | std::ios::trunc);
                    if (!static_push_log_file.is_open()) {
                        Error("Failed to open static push log file: " + filepath);
                        static_test_mode = false;
                    } else {
                        Print("=== STATIC TEST MODE ENABLED ===");
                        Print("Static push log file: " + filepath);
                        Print("Countdown: " + std::to_string((float)static_countdown) + "s");
                        Print("Data collection: " + std::to_string((float)static_data_collection) + "s");
                        Print("Interval: " + std::to_string((float)static_interval) + "s");
                        
                        // Initialize JSON structure
                        static_push_log_file << "{\n  \"static_push_data\": [\n";
                        static_push_log_file.flush();
                        is_first_static_log_entry = true;
                        
                        // Initialize phase
                        static_phase = HOLDING;
                        static_phase_start_time = 0.0; // Will be set in first Tick
                    }
                }
            } catch (const std::exception& e) {
                Error("Exception during static test mode initialization: " + std::string(e.what()));
                static_test_mode = false;
                if (static_push_log_file.is_open()) {
                    static_push_log_file.close();
                }
            }
        }
        else{
            try
            {
                std::string scriptPath = __FILE__;
                std::string scriptDirectory = scriptPath.substr(0, scriptPath.find_last_of("/\\"));
                std::string resultsDir = scriptDirectory + "/results/deviance_logs/" + std::string(prediction_model);

                // Create directory if it doesn't exist
                if (!std::filesystem::exists(resultsDir))
                {
                    if (!std::filesystem::create_directories(resultsDir))
                    {
                        Error("Failed to create results directory: " + resultsDir);
                       
                    }
                }

            
                // Get current time for unique filename
                auto now = std::chrono::system_clock::now();
                auto now_c = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss_time;
                ss_time << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M");
                std::string time_stamp_str = ss_time.str();

                std::string filepath = resultsDir + "/deviance_log_" + time_stamp_str + "position_control" + ".json";

                deviance_log_file.open(filepath, std::ios::out | std::ios::trunc);
                if (!deviance_log_file.is_open())
                {
                    Error("Failed to open deviance log file: " + filepath);
                
                }
                else
                {
                    Debug("Opened deviance log file: " + filepath);
                    // Initialize with a valid empty JSON array structure
                    deviance_log_file << "{\n\"deviance_data\" : [\n{ \n]}\n";
                    deviance_log_file.flush();          // Ensure it's written
                    is_first_deviance_log_entry = true; // This flag now means "is the *next logged entry* the first one in the array?"
                }
            
            }
            catch (const std::exception &e)
            {
                Error("Exception during deviance log initialization: " + std::string(e.what()));
           
                if (deviance_log_file.is_open())
                {
                    deviance_log_file.close();
                }
            }
        }
        
 
    }

    
    void Tick()
    {   
        
       
        // First check if required inputs are connected
        if (!present_current.connected() || !present_position.connected()) {
            Error("Present current and present position must be connected");
            return;
        }
        
        // Check for goal position input
        if (!goal_position_in.connected()) {
            Error("GoalPositionIn must be connected. Use GoalSetter module to provide goal positions.");
            return;
        }
        
        // Check if goal has changed - this indicates GoalSetter sent a new goal after delay
        

        if (static_test_mode) {
            // Initialize phase start time and capture initial position on first tick
            if (GetTick() == 1) {
                if (static_initial_position.size() != present_position.size()) {
                    Error("static_initial_position size mismatch: " + std::to_string(static_initial_position.size()) + 
                          " vs " + std::to_string(present_position.size()));
                    return;
                }
                static_phase_start_time = GetTime();
                // Store the initial position to maintain throughout all cycles
                static_initial_position.copy(present_position);
                Print("\n=== STATIC TEST MODE ACTIVE ===");
                Print("Motors will hold current position: " + static_initial_position.json());
                Print("\n=== CYCLE " + std::to_string(static_test_cycle + 1) + " ===");
                Print("HOLDING position for " + std::to_string((float)static_countdown) + " seconds...");
            }
            
            double elapsed = GetTime() - static_phase_start_time;
            
            // State machine for static test phases
            switch (static_phase) {
                case HOLDING:
                    if (elapsed >= (float)static_countdown) {
                        static_phase = READY_TO_PUSH;
                        static_phase_start_time = GetTime();
                        Print("\n*** READY: Apply push NOW for " + std::to_string((float)static_data_collection) + " seconds! ***");
                        Print("(Try different push strengths and directions)");
                    }
                    break;
                    
                case READY_TO_PUSH:
                    // Transition immediately to pushing phase
                    static_phase = PUSHING;
                    static_phase_start_time = GetTime();
                    Print("LOGGING push data...");
                    break;
                    
                case PUSHING:
                    // Log data during push
                    LogStaticPushData();
                    
                    if (elapsed >= (float)static_data_collection) {
                        static_phase = RESTING;
                        static_phase_start_time = GetTime();
                        Print("Push phase complete. Resting for " + std::to_string((float)static_interval) + " seconds...");
                    }
                    break;
                    
                case RESTING:
                    if (elapsed >= (float)static_interval) {
                        // Start new cycle
                        static_test_cycle++;
                        static_phase = HOLDING;
                        static_phase_start_time = GetTime();
                        Print("\n=== CYCLE " + std::to_string(static_test_cycle + 1) + " ===");
                        Print("HOLDING position for " + std::to_string((float)static_countdown) + " seconds...");
                    }
                    break;
            }
            
            
            
           
            return;
        }
       
        
        ReachedGoal(present_position, goal_position_in, reached_goal, position_margin);
        approximating_goal = ApproximatingGoal(present_position, goal_position_in, position_margin);

        // Skip processing on first tick, just initialize start time
        if (GetTick() <= 2) {
            transition_start_time[transition].set(GetTime());
            if (start_position_in.connected() && first_start_position) {
               
                for (int i = 0; i < servos_to_test.size(); i++) {
                    starting_positions(transition, i) = start_position_in(servos_to_test(i));
                }
            }
            
            return;
        }

        // Handle new goal arrival - this means GoalSetter has sent a new goal after the delay
        if (goal_changed && transition < number_transitions) {
            Debug("New goal detected for transition " + std::to_string(transition));
            
            // Capture starting position for this new transition
            for (int i = 0; i < servos_to_test.size(); i++) {
                int servo_idx = servos_to_test(i);
                starting_positions(transition, i) = present_position(servo_idx);
                transition_start_time(transition, i) = GetTime();
            }
            
            // Store the predicted current for this transition
            for (int i = 0; i < servos_to_test.size(); i++) {
                int servo_idx = servos_to_test(i);
                predicted_current_history(transition, i) = model_prediction(servo_idx);
            }
        }

        // Log data BEFORE incrementing transition - captures current transition's data
        if (GetTick() > 2) {
            LogDevianceData();
        }

        // Check if all servos reached their goals for current transition
        if (reached_goal.sum() == servos_to_test.size()) {
            // Record final current differences for this transition (only once)
            for (int i = 0; i < servos_to_test.size(); i++) {
                int servo_idx = servos_to_test(i);
                if (current_differences(transition, i) == 0) {
                    current_differences(transition, i) = present_current[servo_idx] - initial_currents(transition, i);
                }
            }
            
            // If goal has changed, it means GoalSetter has sent new goal after delay
            // Now we can increment transition counter
            if (goal_changed && transition < number_transitions) {
                Debug("Transition " + std::to_string(transition) + " complete. Moving to transition " + std::to_string(transition + 1));
                transition++;
                
                // Reset reached_goal flags for new transition
                for (int i = 0; i < servos_to_test.size(); i++) {
                    int servo_idx = servos_to_test(i);
                    reached_goal(servo_idx) = 0;
                    approximating_goal(servo_idx) = 0;
                }
            }
            
            // Check if all transitions are complete
            if (transition >= number_transitions) {
                Notify(msg_terminate, "All transitions finished");
                return;
            }
        }

   
        
        // Process servo movements and timeouts
        for (int i = 0; i < servos_to_test.size(); i++) {
            int servo_idx = servos_to_test(i);
            double current_time = GetTime();
            bool timeout = current_time - transition_start_time(transition, servo_idx) > TIMEOUT_DURATION;

            // If servo hasn't reached goal yet
            if (reached_goal(servo_idx) == 0) {
                // Calculate prediction error if we've reached at least 1% of the distance to goal
                // and we haven't already calculated it for this servo in this transition
                if (timeout_occurred(transition, i) == 1 && prediction_error(transition, i) == 0) {
                    // Calculate how far we've moved toward the goal
                    float start_pos = starting_positions(transition, i);
                    float goal_pos = goal_position_in(servo_idx);
                    float current_pos = present_position(servo_idx);
                    float total_distance = abs(goal_pos - start_pos);
                    float traveled_distance = abs(current_pos - start_pos);
                    
                    // Check if we've traveled at least 1% of the distance to goal
                    if (total_distance > 0 && traveled_distance >= 0.01 * total_distance) {
                        // Calculate prediction error (absolute difference between predicted and actual current)
                        float predicted = predicted_current_history(transition, i);
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
       
    


    ~TestMapping()
    {

        // Finalize and close deviance log file
        if ( deviance_log_file.is_open()) {
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
        
        // Finalize and close static push log file
        if (static_test_mode && static_push_log_file.is_open()) {
            static_push_log_file << "\n  ]\n}\n"; // Close JSON structure
            static_push_log_file.close();
            Print("Closed static push log file. Total cycles: " + std::to_string(static_test_cycle));
        }
    }






}; // End of TestMapping class

INSTALL_CLASS(TestMapping)


