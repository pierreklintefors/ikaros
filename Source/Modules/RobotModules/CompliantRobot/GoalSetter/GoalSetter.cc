#include "ikaros.h"
#include <random>
#include <chrono>

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
    std::chrono::steady_clock::time_point reached_goal_time_point;
    std::chrono::steady_clock::time_point final_transition_time_point;

    int time_before_restart = 5; // seconds before restarting transitions
    
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

    void ReachedGoal(matrix present_position, matrix goal_positions, matrix reached_goal, int margin)
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

    //Bind override goal position
    Bind(override_goal_position, "OVERRIDE_GOAL_POSITION");

    if (!goal_position_in.connected() || !one_cycle)
    {
        planned_positions = RandomisePositions(num_transitions, min_limit_position, max_limit_position, robot_type);
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
    Debug("GoalSetter: Goal pos in: " + goal_position_in.json());
    Debug("GoalSetter: Override goal pos: " + override_goal_position.json());
    if (initialising_tick && present_position.sum() > 0)
    {
        initialising_tick = false;
        start_position.copy(present_position);   
        
    }
    if (override_goal_position.sum() > 0)
    {
        Debug("GoalSetter: Override goal position set, using it for next transition");
        goal_position.copy(override_goal_position);
        reached_goal.set(0); // Reset reached_goal for override transition
    }
    // For one_cycle mode, ensure we have a valid goal position
    else if (one_cycle && goal_position_in.connected() && !going_to_neutral)
    {
        goal_position.copy(goal_position_in);
    }

    if (goal_position.sum() > 0)
        ReachedGoal(present_position, goal_position, reached_goal, position_margin);

   
    if (reached_goal.sum() == num_servos && transition < num_transitions && goal_position.sum() > 0)
    {
        Debug("GoalSetter: Reached goal starting new transition");
        // Take time to wait for transition delay
        if (get_time_after_transition && goal_position.sum() > 0)
        {
            reached_goal_time_point = std::chrono::steady_clock::now();
            get_time_after_transition = false;
        }


        if (std::chrono::steady_clock::now() - reached_goal_time_point > std::chrono::seconds(transition_delay.as_int()) || override_goal_position.sum() > 0)
        {
            if (override_goal_position.sum() > 0)
            {
                Debug("GoalSetter: Transitioning to override goal position");
                goal_position.copy(override_goal_position);
                reached_goal.set(0); // Reset reached_goal for override transition
                
            }  
            else if (transition < num_transitions && !one_cycle)
            {
                Debug("GoalSetter: Transitioning to next goal position");
                // Reset reached_goal for the next transition
                transition++;
                start_position.copy(present_position);
                goal_position.copy(planned_positions[transition]);
                reached_goal.set(0);
            }
            else if (one_cycle && goal_position_in.connected())
            {
                // Alternate between goal_position_in and neutral position (180)
                start_position.copy(present_position);
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
                reached_goal.set(0);
            }

            get_time_after_transition = true; // TO get the time for when next transition is finished
        }
    }
    //This does not work yet. Needs somthing like this for avoidance mode
    else if (override_goal_position.connected() && override_goal_position.sum() > 0)
    {
            Debug("GoalSetter: Goal position overriden, updating goal position");
            goal_position.copy(override_goal_position);
    }
    
    

    else if (reached_goal.sum() == num_servos && transition == num_transitions)
    {
        Print("All goals reached. Starting over with same transitions in " + std::to_string(time_before_restart) + " seconds");
        if (get_time_after_all_transitions)
        {
            final_transition_time_point = std::chrono::steady_clock::now();
            get_time_after_all_transitions = false;
        }
        auto now = std::chrono::steady_clock::now();

        if (std::chrono::steady_clock::now() - final_transition_time_point > std::chrono::seconds(time_before_restart) || override_goal_position.sum() > 0)
        {
            if (override_goal_position.sum() > 0)
            {
                Debug("GoalSetter: Restarting with override goal position");
                goal_position.copy(override_goal_position);
            }
            else
            {
                Debug("GoalSetter: Restarting with planned positions");
                goal_position.copy(planned_positions[0]);
            }

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
    Debug("GoalSetter: Time since last transition: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - reached_goal_time_point).count() / 1000.0) + " seconds");

   }
};

INSTALL_CLASS(GoalSetter);


