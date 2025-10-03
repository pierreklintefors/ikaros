#include "ikaros.h"
#include <random>

using namespace ikaros;

class GoalSetter: public Module
{
    //parameters
    parameter position_margin;
    matrix min_limit_position;
    matrix max_limit_position;
    parameter robot_type;
    parameter num_transitions;
    parameter num_servos;
    parameter transition_delay;
    parameter one_cycle;
    parameter static_test_mode;

    //inputs
    matrix present_position;
    matrix goal_position_in;
    matrix override_goal_position;
   

    //outputs
    matrix goal_position;
    matrix start_position;
   


    //internal
    matrix reached_goal;
    matrix planned_positions;
    matrix previous_goal_position_in;
    std::random_device rd;
    int transition;
    bool initialising_tick;
    bool going_to_neutral;
    bool get_time_after_transition;
    bool get_time_after_all_transitions;
    double reached_goal_time_point;
    double final_transition_time_point;

    int time_before_restart = 5; // seconds before restarting transitions
    
    // Function to print a progress bar of the current transition
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
    
    matrix RandomisePositions(int num_transitions, matrix min_limits, matrix max_limits, std::string robotType)
    {
        // Initialize the random number generator with the seed
        std::mt19937 gen(rd());
        int servos_to_control = (robotType == "Torso") ? 2 : 12;
        matrix generated_positions(num_transitions, present_position.size());
        generated_positions.set(180); // Neutral position
        // set pupil servos to 12 of all rows
        for (int i = 0; i < generated_positions.rows(); i++)
        {
            generated_positions(i, 4) = 12; //Neutral position for pupils
            generated_positions(i, 5) = 12; //Neutral position for pupils
        }

        for (int i = 0; i < num_transitions; i++)
        {
            for (int j = 0; j < servos_to_control; j++)
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

    void ReachedGoalCheck(matrix present_position, matrix goal_positions, matrix reached_goal, int margin)
    {
        
        for (int i = 0; i < num_servos; i++)
            {
           

            if (reached_goal(i) == 0 &&
                abs(present_position(i) - goal_positions(i)) < margin)
            {

                reached_goal(i) = 1;

            }
        }
    }

   void Init()
   {

    //Bind inputs and outputs
    Bind(present_position, "PRESENT_POSITION");
    
    

    Bind(goal_position, "GOAL_POSITION");
    Bind(start_position, "START_POSITION");
    Bind(goal_position_in, "GOAL_POSITION_IN");


    //Bind parameters
    Bind(min_limit_position, "MinLimitPosition");
    Bind(max_limit_position, "MaxLimitPosition");
    Bind(num_servos, "NumServos");
    Bind(one_cycle, "OneCycle");
    Bind(robot_type, "RobotType");
    Bind(num_transitions, "NumTransitions");
    Bind(position_margin, "PositionMargin");
    Bind(transition_delay, "TransitionDelay");
    Bind(static_test_mode, "StaticTestMode");

    //Bind override goal position
    Bind(override_goal_position, "OVERRIDE_GOAL_POSITION");

    planned_positions = RandomisePositions(num_transitions, min_limit_position, max_limit_position, robot_type);

    if (!goal_position_in.connected() || !one_cycle)
    {
        
        goal_position.copy(planned_positions[0]);
    }
    else if (one_cycle)
    {     
        goal_position.copy(goal_position_in);
    }
    reached_goal = matrix(present_position.size());
    reached_goal.set(0);
    initialising_tick = true;
    transition = 0;
    going_to_neutral = false;
    get_time_after_transition = true;
    get_time_after_all_transitions = true;

    
   }
   


   void Tick()
   {
    // In static test mode, just hold current position and don't process transitions
    if (static_test_mode) {
        // On first tick, capture and hold the current position
        if (GetTick() == 1 && present_position.sum() > 0) {
            goal_position.copy(present_position);
            start_position.copy(present_position);
            Debug("GoalSetter: Static test mode - holding position at " + goal_position.json());
        }
        // Keep sending the same goal position every tick
        return;
    }
    
    // Helper function to check if override is meaningful (any non-zero value)
    auto has_override = [this]() -> bool {
        if (!override_goal_position.connected() || override_goal_position.size() == 0) {
            Debug("GoalSetter: Override not connected or empty");
            return false;
        }
        for (int i = 0; i < override_goal_position.size(); i++) {
            if (std::abs(override_goal_position[i]) > 0.1) { // Small threshold to avoid floating point issues
                Debug("GoalSetter: Override detected - motor " + std::to_string(i) + " = " + std::to_string(override_goal_position[i]));
                return true;
            }
        }
        Debug("GoalSetter: Override all values below threshold");
        return false;
    };

    Debug("GoalSetter: Goal pos in: " + goal_position_in.json());
    Debug("GoalSetter: Override goal pos: " + override_goal_position.json());
    
    if (initialising_tick && present_position.sum() > 0)
    {
        initialising_tick = false;
        start_position.copy(present_position);   
    }
    
    bool override_active = has_override();
    
    // Handle override goal position first (highest priority)
    if (override_active)
    {
        Debug("GoalSetter: Override goal position set, using it for next transition");
        goal_position.copy(override_goal_position);
        reached_goal.set(0); // Reset reached_goal for override transition
        return; // Exit early when override is active
    }
    // For one_cycle mode, ensure we have a valid goal position
    else if (one_cycle && goal_position_in.connected() && !going_to_neutral)
    {
        goal_position.copy(goal_position_in);
    }

    // Always check if goal is reached when we have valid goal and present positions
    if (goal_position.size() > 0 && present_position.size() > 0)
        ReachedGoalCheck(present_position, goal_position, reached_goal, position_margin);

   
    if (reached_goal.sum() == num_servos && transition < num_transitions && goal_position.size() > 0)
    {
        Debug("GoalSetter: Reached goal starting new transition");
        // Take time to wait for transition delay
        if (get_time_after_transition && goal_position.size() > 0)
        {
            reached_goal_time_point = GetTime();
            get_time_after_transition = false;
        }


        if (GetTime() - reached_goal_time_point >= transition_delay.as_int())
        {
            PrintProgressBar(transition, num_transitions);
            
            if (transition < num_transitions && !one_cycle)
            {
                Debug("GoalSetter: Transitioning to next goal position");

                goal_position.copy(planned_positions[transition]);
                
            }
            else if (one_cycle && goal_position_in.connected())
            {
                // Alternate between goal_position_in and neutral position (180)
                
                if (!going_to_neutral)
                {
                    // Currently at goal position, now go to neutral
                    for (int i = 0; i < num_servos; i++)
                    {
                        goal_position(i) = 180;
                    }
                    going_to_neutral = true;
                }
                else
                {
                    // Currently at neutral, now go to goal position (update from input)
                    goal_position.copy(goal_position_in);
                    going_to_neutral = false;
                }
                start_position.copy(present_position);
                reached_goal.set(0);
                transition++;
            }

            get_time_after_transition = true; // TO get the time for when next transition is finished
        }
    }
    //This section is now handled at the top with early return
    // else if (override_goal_position.connected() && has_override())
    // {
    //         Debug("GoalSetter: Goal position overriden, updating goal position");
    //         goal_position.copy(override_goal_position);
    // }
    
    

    else if (reached_goal.sum() == num_servos && transition == num_transitions)
    {
        Print("All goals reached. Starting over with same transitions in " + std::to_string(time_before_restart) + " seconds");
        if (get_time_after_all_transitions)
        {
            final_transition_time_point = GetTime();
            get_time_after_all_transitions = false;
        }

        if (GetTime() - final_transition_time_point > time_before_restart)
        {
            Debug("GoalSetter: Restarting with planned positions");
            goal_position.copy(planned_positions[0]);

            get_time_after_all_transitions = true; // Reset for next cycle
            initialising_tick = true;
            transition = 0;
            reached_goal.set(0);
            start_position.copy(present_position);
            if (one_cycle)
            {
                goal_position.copy(goal_position_in);
            }
          
        }
    }
    Debug("GoalSetter: Time since last transition: " + std::to_string(GetTime() - reached_goal_time_point) + " seconds");

  
   }

};

INSTALL_CLASS(GoalSetter);


