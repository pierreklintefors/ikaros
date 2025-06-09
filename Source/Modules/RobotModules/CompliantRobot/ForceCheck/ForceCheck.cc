#include "ikaros.h"
//#include "dynamixel_sdk.h"

using namespace ikaros;

class ForceCheck: public Module
{

    //parameters
    
    parameter force_change_rate;
    
    //inputs
    matrix present_current;
    matrix current_limit;
    matrix present_position;// assumes degrees
    matrix goal_position_in;
    matrix start_position;
    matrix allowed_deviance;

    //outputs
    matrix goal_position_out;
    matrix current_prediction;
    matrix force_output;
    matrix deviation;
    matrix led_intensity;
    matrix led_color_eyes;
    matrix led_color_mouth;
    
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
   
    bool firstTick;
    bool obstacle;
    long obstacle_time;
    bool goal_reached;
    long start_time;
    int goal_time_out;
    bool force_output_tapped;
    long force_output_tapped_time;

    

   

    double Tapering(double error, double threshold)
    {
        if (abs(error) < threshold)
            return sin(M_PI_2)*(abs(error)/threshold);
        else
            return 1;
    }
    
    // Calculate RGB color based on deviance ratio (0.0 to 1.0+)
    // 0.0 = white, 0.33 = yellow, 0.66 = orange, 1.0+ = red
    matrix CalculateDevianceColor(double deviance_ratio)
    {
        matrix color(3); // RGB
        Debug("ForceCheck: Deviance ratio: " + std::to_string(deviance_ratio));
        deviance_ratio = clip(deviance_ratio, 0.0, 1.0); // Ensure ratio is between 0 and 1
        
        // Define color transition points (ratio, R, G, B)
        const double color_points[][4] = {
            {0.0,   1.0, 1.0, 1.0},  // White
            {0.20,  0.9, 1.0, 0.9},  // Very Light Cyan
            {0.40,  0.8, 1.0, 0.8},  // Light Cyan
            {0.50,  0.9, 1.0, 0.6},  // Cyan-Yellow Transition
            {0.55,  1.0, 1.0, 0.5},  // Light Yellow-Cyan
            {0.60,  1.0, 1.0, 0.4},  // Light Yellow
            {0.62,  1.0, 1.0, 0.3},  // Yellow-Cyan
            {0.64,  1.0, 1.0, 0.2},  // Yellow Transition
            {0.66,  1.0, 1.0, 0.1},  // Bright Yellow
            {0.68,  1.0, 1.0, 0.0},  // Pure Yellow
            {0.70,  1.0, 0.8, 0.0},  // Yellow-Orange
            {0.72,  1.0, 0.6, 0.0},  // Light Orange
            {0.74,  1.0, 0.5, 0.0},  // Orange
            {0.76,  1.0, 0.4, 0.0},  // Dark Orange
            {0.78,  1.0, 0.3, 0.0},  // Deep Orange
            {0.80,  1.0, 0.2, 0.0},  // Very Deep Orange
            {0.85,  1.0, 0.1, 0.0},  // Orange-Red
            {0.90,  1.0, 0.05, 0.0}, // Red-Orange
            {0.95,  1.0, 0.02, 0.0}, // Almost Red
            {1.0,   1.0, 0.0, 0.0}   // Pure Red
        };
        const int num_points = sizeof(color_points) / sizeof(color_points[0]);
        
        // Find the appropriate color segment
        for (int i = 0; i < num_points - 1; i++) {
            if (deviance_ratio <= color_points[i + 1][0]) {
                // Calculate interpolation factor
                double t = (deviance_ratio - color_points[i][0]) / 
                          (color_points[i + 1][0] - color_points[i][0]);
                
                // Interpolate each color component
                for (int c = 0; c < 3; c++) {
                    color(c) = color_points[i][c + 1] + 
                             t * (color_points[i + 1][c + 1] - color_points[i][c + 1]);
                }
                break;
            }
        }
        
        return color;
    }

    matrix PredictionDeviance(matrix current_prediction, matrix current_output){
        matrix deviance(current_prediction.size());
        for (int i = 0; i < current_prediction.size(); i++) {
            int prediction = current_prediction[i];
            int output = current_output[i];
            deviance[i] = abs(prediction) - abs(output);
        }
        return deviance;
    }

    
    

    bool GoalReached(matrix present_position, matrix goal_position, int margin)
    {
    
        for (int i = 0; i < present_position.size(); i++) {
            int position = present_position[i];
            int goal = goal_position[i];

            if (abs(position - goal) > margin)
                return false;
        }
        return true;
    }

    // Returns a matrix where each element indicates if the corresponding motor
    // has started its transition from its start_position.
    // 1 for started, 0 for not started.
    matrix StartedTransition(const matrix& present_position, const matrix& start_position, int margin)
    {
        if (present_position.size() == 0 || present_position.size() != start_position.size())
        {
            Warning("ForceCheck: StartedTransition - Input matrix size mismatch or empty. Returning empty matrix.");
            return matrix(0); 
        }

        matrix transition_status(present_position.size());
        for (int i = 0; i < start_position.size(); i++){
            if (abs(start_position(i) - present_position(i)) < margin || present_current(i)<10)
                transition_status(i) = 0.0; // Not started
            else
                transition_status(i) = 1.0; // Started
        }
        Debug("ForceCheck: StartedTransition status matrix: " + transition_status.json());
        return transition_status;
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
        Bind(allowed_deviance, "AllowedDeviance");
        Bind(force_change_rate, "ForceChangeRate");
        Bind(led_intensity, "LedIntensity");
        Bind(led_color_eyes, "LedColorEyes");
        Bind(led_color_mouth, "LedColorMouth");
        force_output.set(100);
    

        start_time = std::time(nullptr);
        goal_time_out = 5;
        firstTick = true;
        obstacle = false;
        obstacle_time =0;
        goal_reached = false;
        previous_position.set_name("PreviousPosition");
        position_margin = 3;
        led_intensity.set(0.5);
        led_color_eyes.set(1);
        led_color_mouth.set(1);

        //12x3 (RBG) LED in eyes
        red_color_RGB = {1,0.3,0.3};
        force_output_tapped = false;
        force_output_tapped_time = 0;

    }

    void Tick()
    {   
        const double PROPORTIONAL_DEVIATION_K = 0.5; // Factor for proportional force reduction
        if (present_position.connected() && present_position.size() > 0) {
            previous_position.copy(present_position.last());
        } else if (present_position.connected()) { // Connected but size 0
            previous_position.resize(0);
        }
        // If present_position is not connected, previous_position doesn't get updated from it.

        if (firstTick){
            goal_position_out.copy(goal_position_in);
            if (present_position.connected() && present_position.size() > 0) { // Ensure input is connected and valid
                previous_position.copy(present_position); 
            } else if (present_position.connected()) { // Connected but size 0
                 previous_position.resize(0); // Match size if 0
            }
            // If not connected, previous_position remains uninitialized or as per its default constructor state.
            // This is handled by checks like previous_position.size() > i later.
        }

        if (!present_current.connected() || !current_prediction.connected()) {
            Warning("ForceCheck: PresentCurrent or CurrentPrediction not connected. Skipping deviation calculation.");
            // Decide if module should return or operate with default/no deviation
            if (firstTick) firstTick = false; // Still advance firstTick
            // previous_position update should still happen if present_position is connected
            if (present_position.connected() && present_position.size() > 0) {
                 previous_position.copy(present_position.last());
            } else if (present_position.connected()) {
                 previous_position.resize(0);
            }
            return; 
        }
        deviation.copy(current_prediction);      // Initialize deviation with current_prediction's values
        deviation.subtract(present_current);     // Subtract present_current from deviation in-place
                                                 // current_prediction should remain unmodified by this specific operation.
        
       


        if (present_position.connected() && goal_position_in.connected() && !firstTick){ // current_prediction and present_current checked above

            Debug("ForceCheck: Deviation: " + deviation.json());
            Debug("ForceCheck: Present current: " + present_current.json());
            Debug("ForceCheck: Current prediction: " + current_prediction.json());
            Debug("ForceCheck: Goal position: " + goal_position_in.json());
            Debug("ForceCheck: Start position: " + start_position.json());
            Debug("ForceCheck: Present position: " + present_position.json());
            Debug("ForceCheck: Previous position: " + previous_position.json());
            Debug("ForceCheck: Force output: " + force_output.json());

            
            
            // Calculate deviance ratio and color
            double deviance_ratio = 0.0;
            double intensity = 0.5; // Default intensity for no deviance
            
            //find highest deviance
            double highest_deviance = 0.0;
            int highest_deviance_index = 0;
            for (int i = 0; i < deviation.size(); i++) {
                if (abs(deviation[i]) > highest_deviance && started_transition[i] == 1) {
                    highest_deviance = abs(deviation[i]);
                    highest_deviance_index = i;
                }
            }
            deviance_ratio = highest_deviance / allowed_deviance[highest_deviance_index];
            intensity = deviance_ratio;
            intensity = clip(intensity, 0.5, 1.0);
         

            //if force_output_tapped is true, set force_output to 0
            if (force_output_tapped){
                force_output.set(0);
                Debug("ForceCheck: Force output tapped - setting force to 0");
            }
            if (!force_output_tapped){
                force_output.set(100);
                Debug("ForceCheck: Force output not tapped - setting force to 100");
            }

            // Check if we need to set force_output_tapped based on deviance
            if (deviance_ratio > 0.95 && !force_output_tapped){
                force_output_tapped = true;
                force_output_tapped_time = GetTime();
                Debug("ForceCheck: Force output tapped triggered by high deviance");
            }

            // Reset force_output_tapped after timeout
            if (force_output_tapped && (GetTime() - force_output_tapped_time > 2)){
                force_output_tapped = false;
                force_output_tapped_time = 0;
                Debug("ForceCheck: Force output tap timeout - resetting");
            }
        
            Debug("ForceCheck: Force output tapped time: " + std::to_string(force_output_tapped_time));
          

            started_transition = StartedTransition(present_position, start_position, position_margin);
            goal_reached = GoalReached(present_position, goal_position_in, position_margin);
            // Get the color for current deviance level
            matrix current_color = CalculateDevianceColor(deviance_ratio);
            
            if (started_transition.sum() > 0){
                // Set LED intensity
                led_intensity.set(intensity);
                
                // Apply color to all eye LEDs (12 LEDs, 3 RGB values each)
                for (int i = 0; i < 12; i++) {
                    led_color_eyes(0, i) = current_color(0); // Red
                    led_color_eyes(1, i) = current_color(1); // Green
                    led_color_eyes(2, i) = current_color(2); // Blue
                }
                
                // Apply color to all mouth LEDs (8 LEDs, 3 RGB values each)
                for (int i = 0; i < 8; i++) {
                    led_color_mouth(0, i) = current_color(0); // Red
                    led_color_mouth(1, i) = current_color(1); // Green
                    led_color_mouth(2, i) = current_color(2); // Blue
                }
                
                Debug("ForceCheck: Deviance ratio: " + std::to_string(deviance_ratio) + 
                    ", Intensity: " + std::to_string(intensity) + 
                    ", Color RGB: [" + std::to_string(current_color(0)) + 
                    ", " + std::to_string(current_color(1)) + 
                    ", " + std::to_string(current_color(2)) + "]");
            } else {
                led_intensity.set(0.5);
                led_color_eyes.set(1);
                led_color_mouth.set(1);
            }
            
        
        } else {
            if (!present_position.connected()) Warning("ForceCheck: PresentPosition not connected.");
            if (!goal_position_in.connected()) Warning("ForceCheck: GoalPositionIn not connected.");
            // If critical inputs aren't connected, module can't do much.
            // Force output will remain as is.
        }
       
        firstTick=false;
        
        
       
    }


};


INSTALL_CLASS(ForceCheck);

