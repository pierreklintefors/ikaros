#include "ikaros.h"
#include <random> // Include the header file that defines the 'Random' function
#include <fstream>
#include <vector>
#include <memory>
#include "json.hpp"
#include <string>
#include <algorithm>
#include <chrono>

using json = nlohmann::json;
using namespace ikaros;
using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

class CurrentPositionMapping: public Module
{
  
    matrix present_position;
    matrix present_current;
    matrix position_transitions;
    matrix previous_position;
    matrix start_position;
    matrix gyro;
    matrix accel;
    matrix eulerAngles;
    
    matrix goal_current;
    matrix goal_position_out;
    matrix current_controlled_servos;
    matrix max_present_current;
    matrix min_moving_current;
    matrix min_torque_current;
    matrix overshot_goal;
    matrix overshot_goal_temp;
    std::vector<std::shared_ptr<std::vector<float>>> moving_trajectory;
    std::vector<std::shared_ptr<std::vector<float>>> current_history;
    std::vector<std::shared_ptr<std::vector<float>>> gyro_history;
    std::vector<std::shared_ptr<std::vector<float>>> accel_history;
    std::vector<std::shared_ptr<std::vector<float>>> angles_history;
    matrix approximating_goal;
    matrix reached_goal;
   
 

    parameter num_transitions;
    parameter min_limits;
    parameter max_limits;
    parameter robotType;


    std::random_device rd;
    
    int number_transitions; //Will be used to add starting and ending position
    int position_margin = 3;
    int transition = 0;
    int current_increment = 2;
    int starting_current = 30;
    int current_limit = 1700;
    bool find_minimum_torque_current = false;
    matrix minimum_torque_current_found;
    int unique_id;
    bool second_tick = false;
    bool first_start_position = true;

    matrix transition_start_time;
    matrix transition_duration;
    double time_prev_position;
    double position_sampling_interval = 50;
    matrix number_ticks;
 
    matrix SetGoalCurrent(matrix present_current,  int increment, int limit, matrix position, matrix goal_position, int margin){
        matrix current_output(present_current.size());
        Notify(msg_debug, "Inside SetGoalCurrent()");
        if(present_current.size() != current_output.size()){
            std::cout << present_current.size() << std::endl;
            std::cout << current_output.size() << std::endl;
            Notify(msg_fatal_error, "Present current and Goal current must be the same size");
            return -1;
        }
        for (int i = 0; i < current_controlled_servos.size(); i++) {
            if ( goal_current(current_controlled_servos(i)) < limit && abs(goal_position(current_controlled_servos(i))- position(current_controlled_servos(i)))> margin){
                Notify(msg_debug, "Increasing current");
                current_output(current_controlled_servos(i)) = goal_current(current_controlled_servos(i))+ increment;
                          
            }
            else
            {
                current_output(i) = goal_current(i);
            }
        }    
        return current_output;

    }

    matrix RandomisePositions(matrix present_position, int num_transitions, matrix min_limits, matrix max_limits, std::string robotType){
        // Initialize the random number generator with the seed
        std::mt19937 gen(rd());
        int num_columns = (robotType == "Torso") ? 2 : 12;
        matrix goal_positions(num_transitions, present_position.size());
        goal_positions.set(180);
        if (num_columns != min_limits.size() && num_columns != max_limits.size()){
    
            Notify(msg_fatal_error, "Min and Max limits must have the same number of columns as the current controlled servos in the robot type (2 for Torso, 12 for full body)");
            return -1;
        }
        
        if (robotType=="Torso"){
            for (int i = 0; i < num_transitions; i++) {
                for (int j = 0; j < num_columns; j++) {
                    std::uniform_real_distribution<double> distr(min_limits(j), max_limits(j));
                    goal_positions(i, j) = int(distr(gen));
                    if (i == num_transitions-1){
                        goal_positions(i, j) = 180;
                    }
                }
            }
        }
        
    

        return goal_positions;
    }

    void ReachedGoal(matrix present_position, matrix goal_positions, matrix reached_goal, int margin){
        for (int i = 0; i < current_controlled_servos.size(); i++){
            if (reached_goal(current_controlled_servos(i)) == 0 &&
                abs(present_position(current_controlled_servos(i)) - goal_positions(current_controlled_servos(i))) < margin){

                reached_goal(current_controlled_servos(i)) = 1;

                auto now = Clock::now();
                float current_time_ms = std::chrono::duration<float, std::milli>(now.time_since_epoch()).count();
                transition_duration(current_controlled_servos(i)) = current_time_ms - transition_start_time[current_controlled_servos(i)];
                // Reset start time for future transitions
                transition_start_time(current_controlled_servos(i)) = current_time_ms;      

            }
            else if (reached_goal(current_controlled_servos(i)) == 0){
                number_ticks[current_controlled_servos(i)] ++;
            }

        }
    }

    matrix ApproximatingGoal(matrix present_position, matrix previous_position, matrix goal_position, int margin){
        for (int i =0; i < current_controlled_servos.size(); i++){
            //Checking if distance to goal is decreasing
            if (abs(goal_position(current_controlled_servos(i)) - present_position(current_controlled_servos(i))) < 0.97*abs(goal_position(current_controlled_servos(i))-previous_position(current_controlled_servos(i)))){
                Notify(msg_debug, "Approximating Goal");
                approximating_goal(current_controlled_servos(i)) = 1;
                
            }
            else{
                approximating_goal(current_controlled_servos(i)) = 0;
            }
        }
        return approximating_goal;
    }   
    //saving starting position, goal position and current in json file
    void SaveMetricsToJson(matrix goal_positions, matrix start_position, matrix max_current, matrix min_current, matrix min_torque, matrix overshot, matrix time, matrix ticks, std::string robotType, int transition, int unique_id)
    {
        Notify(msg_debug, "Inside SaveMetricsToJson()");
        std:: cout << "Saving positions in json file" << std::endl;
        std::ofstream file;
        std::string scriptPath = __FILE__;
        std::string scriptDirectory = scriptPath.substr(0, scriptPath.find_last_of("/\\"));
        std::string filePath = scriptDirectory + "/data/CurrentPositionMapping" + robotType + ".json";
        
        // Check if file exists
        std::ifstream checkFile(filePath);
        bool fileExists = checkFile.good();
        checkFile.close();
        
        file.open(filePath, std::ios::app);
    
        if (transition==0 && !fileExists){
            file << "{\n";
            file << "\"Mapping\": [\n";
        }
       // If the file exists, remove the last "\n]\n}" and replace it with ","
        else if (fileExists && transition==0){ 
                std::ifstream inFile(filePath);
                std::string fileContent((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
                inFile.close();
    
                // Check if unique_id is already in the file
                while (fileContent.find(std::to_string(unique_id)) != std::string::npos) {
                    unique_id++;
                }

                size_t pos = fileContent.rfind("\n]\n}");
                if (pos != std::string::npos) {
                    fileContent.replace(pos, 4, ",");
                    std::ofstream outFile(filePath);
                    outFile << fileContent;
                    outFile.close();
                }
            }
    
        //Check if unique_id is already in the file
        file << "{\"UniqueID\": [" << unique_id << "], ";
        file << "\"StartingPosition\": [";
        for (int i = 0; i < current_controlled_servos.size(); i++){
            file << int(start_position(current_controlled_servos(i)));
            if (i < current_controlled_servos.size()-1){
                file << ", ";
            }     
        }
        file << "],";
        file << "\"GoalPosition\": [";
        for (int i = 0; i < current_controlled_servos.size(); i++){
            file << int(goal_positions(current_controlled_servos(i)));
            if (i < current_controlled_servos.size()-1){
                file << ", ";
            }
        }
        file << "],";
        file << "\"OvershotGoal\": [";
        for (int i = 0; i < current_controlled_servos.size(); i++){
            file << overshot(current_controlled_servos(i));
            if (i < current_controlled_servos.size()-1){
                file << ", ";
            }
        }
        
        file << "],";
        file << "\"MaxCurrent\": ["; 
        for (int i = 0; i < current_controlled_servos.size(); i++){
            file << int(max_current(current_controlled_servos(i)));
            if (i < current_controlled_servos.size()-1){
                file << ", ";
            }
        }
        
        file << "],";
        file << "\"MinCurrentToMoveFromStart\": ["; 
        for (int i = 0; i < current_controlled_servos.size(); i++){
            file << int(abs(min_current(current_controlled_servos(i))));
            if (i < current_controlled_servos.size()-1){
                file << ", ";
            }
        }
        file << "],";
        file << "\"MinCurrentForTorqueAtGoal\": ["; 
        for (int i = 0; i < current_controlled_servos.size(); i++){
            file << int(abs(min_torque(current_controlled_servos(i))));
            if (i < current_controlled_servos.size()-1){
                file << ", ";
            }
        }
        file << "],";
        file << "\"Time(ms)\": [";
        for (int i = 0; i < current_controlled_servos.size(); i++)
        {
            file << int(abs(time(current_controlled_servos(i))));
            if (i < current_controlled_servos.size() - 1)
            {
                file << ", ";
            }
        }
        file << "],";
        file << "\"Number of Ticks\": [";
        for (int i = 0; i < current_controlled_servos.size(); i++)
        {
            file << int(abs(ticks(current_controlled_servos(i))));
            if (i < current_controlled_servos.size() - 1)
            {
                file << ", ";
            }
        }
        //if transiiton is not the last one, add a comma
        if (transition < number_transitions-1){
            file << "]},";
        }
        else{
            file << "]}\n]\n}";
        }
        
        file << std::endl;
    
        file.close();
    }

    void SaveTrajectory(std::vector<std::shared_ptr<std::vector<float>>> trajectory,
                        std::vector<std::shared_ptr<std::vector<float>>> current,
                        std::vector<std::shared_ptr<std::vector<float>>> gyro,
                        std::vector<std::shared_ptr<std::vector<float>>> accel,
                        std::vector<std::shared_ptr<std::vector<float>>> angles, std::string robotType, int id)
    {
        std::string scriptPath = __FILE__;
        std::string scriptDirectory = scriptPath.substr(0, scriptPath.find_last_of("/\\"));
        std::string filePath = scriptDirectory + "/data/Trajectories" + robotType + ".json";

        // Read existing file content
        nlohmann::json root;
        std::ifstream inFile(filePath);
        if (inFile.good()) {
            inFile >> root;
        }
        inFile.close();

        // Ensure id is unique
        for (const auto& traj : root["Trajectories"]) {
            if (traj["UniqueID"] == id) {
                id++;
            }
        }

        // Create a vector of indices from current_controlled_servos
        std::vector<int> controlled_indices;
        for (int i = 0; i < current_controlled_servos.size(); ++i)
        {
            controlled_indices.push_back(static_cast<int>(current_controlled_servos(i)));
        }

        // Create new trajectory entry
        nlohmann::json newTrajectory;

        for (const auto &g : gyro)
        {
            nlohmann::json jsonPoint3 = nlohmann::json::array();
            for (const auto &value : *g)
            {
                jsonPoint3.push_back(std::round(value * 100.0) / 100.0); // Round to 2 decimal places
            }
            newTrajectory["Gyro"].push_back(jsonPoint3);
        }

        for (const auto &a : accel)
        {
            nlohmann::json jsonPoint4 = nlohmann::json::array();
            for (const auto &value : *a)
            {
                jsonPoint4.push_back(std::round(value * 100.0) / 100.0); // Round to 2 decimal places
            }
            newTrajectory["Accel"].push_back(jsonPoint4);
        }

        for (const auto &ang : angles)
        {
            nlohmann::json jsonPoint5 = nlohmann::json::array();
            for (const auto &value : *ang)
            {
                jsonPoint5.push_back(std::round(value * 100.0) / 100.0); // Round to 2 decimal places
            }
            newTrajectory["Angles"].push_back(jsonPoint5);
        }

        for (const auto &pos : trajectory) {
            nlohmann::json jsonPoint = nlohmann::json::array();
            for (const auto &index : controlled_indices) {
                if (index < pos->size()){                                                                   // Ensure index is within bounds
                    jsonPoint.push_back(std::round((*pos)[index] * 100.0) / 100.0); // Round to 2 decimal places
                }
            }
            newTrajectory["Trajectory"].push_back(jsonPoint);
        }

        for(const auto& c : current){
            nlohmann::json jsonPoint2 = nlohmann::json::array();
            for (const auto &index : controlled_indices) {
                if (index < c->size()){                                                                   // Ensure index is within bounds
                    jsonPoint2.push_back(std::round((*c)[index] * 100.0) / 100.0); // Round to 2 decimal places
                }
            }
            newTrajectory["Current"].push_back(jsonPoint2);
        }

        
        
        newTrajectory["UniqueID"] = id;

        // Add new trajectory to root
        root["Trajectories"].push_back(newTrajectory);

        // Write updated content to file
        std::ofstream outFile(filePath);
        outFile << root.dump(-1); // -1 to compact the JSON
        unique_id = id;
    }

    // FindMinimumTorqueCurrent(present_current, present_position, previous_position, current_limit), returns current and boolean value
    std::pair<matrix, matrix> FindMinimumTorqueCurrent(matrix present_current, matrix present_position, matrix previous_position, int decreasing_step) {
        matrix current_output(present_current.size());
        matrix torque_found_temp;
        torque_found_temp.copy(minimum_torque_current_found);

        for (int i = 0; i < current_controlled_servos.size(); i++) {       
            
            if (torque_found_temp(current_controlled_servos(i)) == 0) {
              
            
                if (int(present_position(current_controlled_servos(i))) == int(previous_position(current_controlled_servos(i))))  {
                    current_output(current_controlled_servos(i)) = abs(present_current(current_controlled_servos(i)))- decreasing_step;
                    //std::cout << "Decreasing goal current to: " << (float)current_output(current_controlled_servos(i)) << " of servo: " << current_controlled_servos(i)+2 << std::endl;

                    if (current_output(current_controlled_servos(i)) < 0) {
                        current_output(current_controlled_servos(i)) = 0;
                        torque_found_temp(current_controlled_servos(i)) = 1;
                        
                    }
                
                } 
                else {
                    current_output(current_controlled_servos(i)) = abs(present_current(current_controlled_servos(i))) + decreasing_step;
                    std::cout << "Minimum Current for torque at goal: " << current_output(i) << std::endl;
                    torque_found_temp(current_controlled_servos(i)) = 1;
                }
            }
        }

        return std::make_pair(current_output, torque_found_temp);
    }

    
    
    //Check if the robot has overshot the goal by comparing the sign of the difference between the starting position and the goal position and the difference between the current position and the goal position
    matrix OvershotGoal (matrix position, matrix goal_position, matrix starting_position, matrix overshot_goal){
        //Matrix of booleans to check if the robot has overshot the goal
        for (int i = 0; i < current_controlled_servos.size(); i++){
            
            if (overshot_goal(i) == false && int((goal_position(current_controlled_servos(i)) - starting_position(current_controlled_servos(i)))) * int((goal_position(current_controlled_servos(i)) - position(current_controlled_servos(i))) ) < 0){
                overshot_goal(i)= true;
            }
        }
        return overshot_goal;
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
    
    void Init()
    {
    
        Bind(present_current, "PresentCurrent");
        Bind(present_position, "PresentPosition");
        Bind(gyro, "GyroData");
        Bind(accel, "AccelData");
        Bind(eulerAngles, "EulerAngles");
        Bind(goal_current, "GoalCurrent");
        Bind(num_transitions, "NumberTransitions");
        Bind(goal_position_out, "GoalPosition");

        Bind(min_limits, "MinLimits");
        Bind(max_limits, "MaxLimits");
        Bind(robotType, "RobotType");
        


        number_transitions = num_transitions.as_int()+1; // Add one to the number of transitions to include the ending position
       
        position_transitions.set_name("PositionTransitions");
        position_transitions = RandomisePositions(present_position, number_transitions, min_limits, max_limits, robotType);
        goal_position_out.copy(position_transitions[0]);
        previous_position.set_name("PreviousPosition");
        previous_position.copy(present_position);
        goal_current.set(starting_current);
        max_present_current.set_name("MaxPresentCurrent");
        max_present_current.copy(present_current);
        min_moving_current.set_name("MinMovingCurrent");
        min_moving_current.copy(present_current);
        min_moving_current.set(1000);
        min_torque_current.set_name("MinTorqueCurrent");
        min_torque_current.copy(present_current);
        min_torque_current.set(1000);
        minimum_torque_current_found.set_name("MinimumTorqueCurrentFound");
        minimum_torque_current_found.copy(present_current);
        minimum_torque_current_found.set(0);
        

        approximating_goal.set_name("ApproximatingGoal");
        approximating_goal.copy(present_position);
        approximating_goal.set(0);
        transition_start_time.set_name("StartedTransitionTime");
        transition_start_time.copy(approximating_goal);
        transition_duration.set_name("TransitionDuration");
        transition_duration.copy(approximating_goal);
        number_ticks.set_name("NumberTicks");



        time_prev_position = std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
        

        std::string robot = robotType;

        if(robot == "Torso"){
            current_controlled_servos.set_name("CurrentControlledServosTorso");
            current_controlled_servos = {0, 1};
            overshot_goal_temp = {false, false};
            reached_goal = {0, 0};
        }
        else{
            current_controlled_servos.set_name("CurrentControlledServosFullBody");
            current_controlled_servos = {0, 1, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18};
            overshot_goal_temp = {false, false, false, false, false, false, false, false, false, false, false, false, false};
            reached_goal = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }

        reached_goal.set_name("ReachedGoal");

        overshot_goal.set_name("OvershotGoal");
        unique_id = GenerateRandomNumber(0, 1000000);
        number_ticks.copy(reached_goal);
    }


    void Tick()
    {   
        
        if (present_current.connected() && present_position.connected() && second_tick){

           

            approximating_goal = ApproximatingGoal(present_position, previous_position, goal_position_out, position_margin);
            
            
        
            if (approximating_goal.sum()>0 && !find_minimum_torque_current) {
                
                if(second_tick && first_start_position){
                    start_position.copy(present_position);
                    auto now = Clock::now();
                    float now_ms = std::chrono::duration<float, std::milli>(now.time_since_epoch()).count();

                    transition_start_time.set(now_ms);
                    first_start_position = false;
                }
                
                moving_trajectory.push_back(std::make_shared<std::vector<float>>(present_position.data_->begin(), present_position.data_->end()));
                
                current_history.push_back(std::make_shared<std::vector<float>>(present_current.data_->begin(), present_current.data_->end()));
                gyro_history.push_back(std::make_shared<std::vector<float>>(gyro.data_->begin(), gyro.data_->end()));
                accel_history.push_back(std::make_shared<std::vector<float>>(accel.data_->begin(), accel.data_->end()));
                angles_history.push_back(std::make_shared<std::vector<float>>(eulerAngles.data_->begin(), eulerAngles.data_->end()));
                

                // Update min_moving_current only at the start of movement for each servo
                for (int i = 0; i < current_controlled_servos.size(); i++) {
                    int servo = current_controlled_servos(i);
                    float abs_current = abs(present_current(servo));
                    
                    // Capture the current when the servo starts moving
                    if (min_moving_current(servo)==1000 && abs_current > 0 && approximating_goal(servo) == 1) {
                        min_moving_current(servo) = abs_current;
                        auto now = Clock::now();
                        float now_ms = std::chrono::duration<float, std::milli>(now.time_since_epoch()).count();
                        transition_start_time(current_controlled_servos(i)) = now_ms;

                       
                    }
                    // Update the max current for each servo
                    if(abs(present_current(current_controlled_servos(i))) > max_present_current(current_controlled_servos(i))){
                    max_present_current(current_controlled_servos(i)) = abs(present_current(current_controlled_servos(i)));
                    }
                }
         
                
            }
            
            else if (find_minimum_torque_current)
            {
                std::pair<matrix, matrix> result = FindMinimumTorqueCurrent(present_current, present_position, previous_position, 1);
                goal_current.copy(result.first);
                minimum_torque_current_found.copy(result.second);
                if(minimum_torque_current_found.sum() == current_controlled_servos.size()){
                    find_minimum_torque_current = false;
                    min_torque_current.copy(present_current);
                    goal_current.add(10);
                    Notify(msg_debug, "Minimum torque current found");
                    
                    
                    //save starting position, goal position and current in json file
                    SaveMetricsToJson(goal_position_out,start_position, max_present_current, min_moving_current, min_torque_current, overshot_goal, transition_duration, number_ticks, robotType, transition, unique_id);
                    transition++;
                    auto now = Clock::now();
                    float current_time_ms = std::chrono::duration<float, std::milli>(now.time_since_epoch()).count();
                    transition_start_time.set(current_time_ms);
                    transition_duration.set(0);
                    
                    PrintProgressBar(transition, number_transitions);

                    if (transition < number_transitions){
                        goal_position_out.copy(position_transitions[transition]);
                        std::cout << "New goal position" << std::endl;
                        goal_current.copy(min_torque_current);
                        min_moving_current.copy(present_current);
                        max_present_current.reset();
                        overshot_goal_temp.reset();
                        min_torque_current.reset();
                        moving_trajectory.push_back(std::make_shared<std::vector<float>>(present_position.data_->begin(), present_position.data_->end()));
                        current_history.push_back(std::make_shared<std::vector<float>>(present_current.data_->begin(), present_current.data_->end()));
                        gyro_history.push_back(std::make_shared<std::vector<float>>(gyro.data_->begin(), gyro.data_->end()));
                        accel_history.push_back(std::make_shared<std::vector<float>>(accel.data_->begin(), accel.data_->end()));
                        angles_history.push_back(std::make_shared<std::vector<float>>(eulerAngles.data_->begin(), eulerAngles.data_->end()));
                        reached_goal.reset();
                        number_ticks.set(0);
  
                        
                    }
                    else{
                        Notify(msg_end_of_file, "All transitions completed");
                        Sleep(1);
                        Notify(msg_terminate, "Shutting down");
                    }
                }
            }
            
            else if (reached_goal.sum() == current_controlled_servos.size()){

                find_minimum_torque_current = true;
                SaveTrajectory(moving_trajectory, current_history, gyro_history, accel_history, angles_history, robotType, unique_id);
                
                // first row of the trajectory matrix is the starting position
                if (!moving_trajectory.empty()) {
                    std::copy(moving_trajectory.front()->begin(), moving_trajectory.front()->end(), start_position.begin());
                }
                

                moving_trajectory.clear();
                current_history.clear();
                gyro_history.clear();
                accel_history.clear();
                angles_history.clear();
                reached_goal.set(0);

                // New start time for the next transition
                //  Save the duration and reset start times for next transition
                auto now = Clock::now();
                float current_time_ms = std::chrono::duration<float, std::milli>(now.time_since_epoch()).count();
                
            }
            else{
                goal_current.copy(SetGoalCurrent(present_current, current_increment, current_limit, present_position, goal_position_out, position_margin));
                overshot_goal.copy(OvershotGoal(present_position, goal_position_out, start_position, overshot_goal_temp));
                overshot_goal_temp.copy(overshot_goal);
                ReachedGoal(present_position, goal_position_out, reached_goal, position_margin);
                // Save present positions into trajectory matrix
                moving_trajectory.push_back(std::make_shared<std::vector<float>>(present_position.data_->begin(), present_position.data_->end()));
                current_history.push_back(std::make_shared<std::vector<float>>(present_current.data_->begin(), present_current.data_->end()));
                gyro_history.push_back(std::make_shared<std::vector<float>>(gyro.data_->begin(), gyro.data_->end()));
                accel_history.push_back(std::make_shared<std::vector<float>>(accel.data_->begin(), accel.data_->end()));
                angles_history.push_back(std::make_shared<std::vector<float>>(eulerAngles.data_->begin(), eulerAngles.data_->end()));
            }

            auto now = Clock::now();
            float now_ms = std::chrono::duration<float, std::milli>(now.time_since_epoch()).count();
            
            if (abs(now_ms - time_prev_position) > position_sampling_interval)
            {
                time_prev_position = now_ms;
                previous_position.copy(present_position);
            }
        }
        
        second_tick = true;
    }

};





INSTALL_CLASS(CurrentPositionMapping)

