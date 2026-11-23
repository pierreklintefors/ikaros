import numpy as np
import mmap
import os
import json
from time import sleep
import sys
import signal
import struct  # Add struct for size calculations
import time # Add this at the top
import pandas as pd  # Import pandas for DataFrame operations
import pickle
import re
import tensorflow as tf
try:
    import posix_ipc
except ImportError as e:
    print(f"Error importing posix_ipc: {e}", file=sys.stderr)
    print(f"Current environment: {os.environ.get('VIRTUAL_ENV')}", file=sys.stderr)
    sys.exit(1)
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, Input, LSTM, GRU, Activation
from keras.regularizers import l2
from keras.optimizers import Adam

import tensorflow as tf
import os # Make sure os is imported if not already

# Optimize TensorFlow for inference performance
tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores

# # Force CPU execution
# tf.config.set_visible_devices([], 'GPU')
# os.environ['TF_DISABLE_METAL'] = '1' # Another way to potentially disable Metal
# print("Attempting to force CPU execution...", file=sys.stderr)


# Define constants for SHM interaction based on PythonScriptCaller.cc
FLOAT_SIZE = 4  # Size of a float in bytes
BOOL_SIZE = 1   # Size of a bool in bytes (Python struct module uses 1 byte for '?')

# Offsets for flags within the flags structure at the beginning of SHM
# These are byte offsets.
CPP_WROTE_INPUT_OFFSET = 0
PYTHON_WROTE_OUTPUT_OFFSET = BOOL_SIZE  # e.g., 1
SHUTDOWN_SIGNAL_OFFSET = BOOL_SIZE * 2 # e.g., 2

# NUM_SERVOS is critical for parsing input and determining output size.
# This script is specific to a 2-servo setup (Tilt and Pan).
# If PythonScriptCaller is configured for a different number of inputs,
# this script would need to adapt or fail.
NUM_SERVOS = 2
EXPECTED_INPUTS_PER_SERVO = 6  # position, dist_to_goal, dist_to_start, goal_pos, start_pos
EXPECTED_BASE_INPUTS = 9 # Gyro(3) + Accel(3) + Angles(3)
# For models with 'current' in the name, we expect one additional input per servo
def get_expected_total_inputs():
    global model_name_suffix
    return EXPECTED_BASE_INPUTS + NUM_SERVOS * EXPECTED_INPUTS_PER_SERVO 


# Define which features to use for each servo model
# TILT_FEATURES = ['GyroX', 'GyroY', 'GyroZ', 'TiltPosition', 'TiltDistToGoal', 'TiltDistToStart']
# PAN_FEATURES = ['PanPosition', 'PanDistToGoal', 'PanDistToStart', 'TiltPosition', 'TiltDistToGoal', 'TiltDistToStart']

# Global constants for file paths and model configuration
directory = os.path.dirname(os.path.abspath(__file__))
servos = ['tilt', 'pan'] 

###################################################################################################################
#############################  Model weights and config filenames. ################################################
###################################################################################################################

#model_name_suffix = 'L1_128_L2_64_position_control_standardised_Relu_mean_squared_error_2Output_DropOut_0.2_l2_0.001_all_data'
#mean_stds_suffix = 'all_data_position_control_mean_std.json'

#model_name_suffix = 'position_control_Output1'
#mean_stds_suffix = 'mean_std_position_control_Output1.json'

#model_name_suffix = 'L1_256_L2_32_position_control_standardised_Relu_mean_squared_error_1Output_DropOut_0.15_l2_0.003_all_data_20250901'
#model_name_suffix = 'rnn_position_control_Output1_TimeSteps5'
#mean_stds_suffix = 'all_data_position_control_mean_std.json'

#Temporal Convolutional Network (TCN) model
# model_name_suffix = 'model_tcn'
#mean_stds_suffix = 'mean_std_tcn.json'

# Temporal MLP model
model_name_suffix = 'temporal_mlp_TimeSteps3_no_IMU_with_current'  # Example for temporal MLP with 3 timesteps and current features
mean_stds_suffix = 'temporal_mlp_mean_std.json'

# For dual-output MLP models, use model names like:
# model_name_suffix = 'mlp_TimeSteps5_current' - for MLP with timesteps and current features
# model_name_suffix = 'mlp_current' - for regular MLP with current features
# model_name_suffix = 'mlp_TimeSteps10' - for MLP with timesteps but no current features

# Initialize scaler objects
pan_feature_scalar_object = None
tilt_feature_scalar_object = None  
pan_y_scalar_object = None
tilt_y_scalar_object = None

if "Output1" in model_name_suffix:
    # Try to load StandardScaler objects first (for RNN models)
    try:
        with open(directory + f'/weights/scalers/scaler_X_pan_position_control_Output1.pkl', 'rb') as f:
            pan_feature_scalar_object = pickle.load(f)
        with open(directory + f'/weights/scalers/scaler_X_tilt_position_control_Output1.pkl', 'rb') as f:
            tilt_feature_scalar_object = pickle.load(f)
        with open(directory + f'/weights/scalers/scaler_y_pan_position_control_Output1.pkl', 'rb') as f:
            pan_y_scalar_object = pickle.load(f)
        with open(directory + f'/weights/scalers/scaler_y_tilt_position_control_Output1.pkl', 'rb') as f:
            tilt_y_scalar_object = pickle.load(f)
        print("✓ Loaded StandardScaler objects for normalization", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"⚠ StandardScaler files not found: {e}. Will use mean/std JSON files.", file=sys.stderr)
        
TILT_FEATURES = []
PAN_FEATURES = []
ALL_FEATURES = []  # For MLP models with dual output

# Check if this is a dual-output MLP model (contains 'mlp' in name)
is_dual_output_mlp = 'mlp' in model_name_suffix.lower()

if is_dual_output_mlp:
    # For MLP models with dual outputs, load combined features from single file
    try:
        features_file = directory + f'/weights/{model_name_suffix}_features.json'
        with open(features_file, 'r') as f:
            ALL_FEATURES = json.load(f)
        print(f"Loaded combined features for dual-output MLP model from: {features_file}", file=sys.stderr)
        print(f"Combined features ({len(ALL_FEATURES)}): {ALL_FEATURES}", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"Combined features JSON file not found: {e}. Using default MLP features.", file=sys.stderr)
        # Default combined features for dual-output MLP
        ALL_FEATURES = ['TiltPosition', 'TiltDistToGoal', 'TiltDistToStart', 
                       'PanPosition', 'PanDistToGoal', 'PanDistToStart']
        if 'current' in model_name_suffix.lower():
            ALL_FEATURES = ['TiltPosition', 'TiltDistToGoal', 'TiltDistToStart', 'TiltCurrent',
                       'PanPosition', 'PanDistToGoal', 'PanDistToStart', 'PanCurrent']
elif "tcn" in model_name_suffix:
    # Define the features based on your TCN model architecture
    # Tilt model expects 12 features, Pan model expects 13 features (based on the error message)
    TILT_FEATURES = [
        'GyroX', 'GyroY', 'GyroZ',
        'AccelX', 'AccelY', 'AccelZ', 
        'AngleX', 'AngleY', 'AngleZ',
        'TiltPosition', 'TiltDistToGoal', 'TiltDistToStart'
    ]  # 12 features total
    
    PAN_FEATURES = [
        'GyroX', 'GyroY', 'GyroZ',
        'AccelX', 'AccelY', 'AccelZ',
        'AngleX', 'AngleY', 'AngleZ', 
        'PanPosition', 'PanDistToGoal', 'PanDistToStart',
        'TiltPosition'  # Pan model also uses tilt position
    ]  # 13 features total
    
    
elif "temporal_mlp" in model_name_suffix:
    # For temporal MLP models, load features from the _features.json files
    try:
        with open(directory + f'/weights/tilt_{model_name_suffix}_features.json', 'r') as f:
            TILT_FEATURES = json.load(f)
        with open(directory + f'/weights/pan_{model_name_suffix}_features.json', 'r') as f:
            PAN_FEATURES = json.load(f)
        print(f"Loaded features from JSON files for temporal MLP models", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"Features JSON files not found: {e}. Using default temporal MLP features.", file=sys.stderr)
        # Use the same features as defined in your training code
        TILT_FEATURES = ['TiltPosition', 'TiltDistToGoal', 'TiltDistToStart']
        PAN_FEATURES = ['PanPosition', 'PanDistToGoal', 'PanDistToStart', 'TiltPosition']
else:
    # Load features from config files for non-TCN models
    for servo in servos:
        if servo == 'tilt':
            with open(directory + f'/weights/{servo}_{model_name_suffix}_config.json', 'r') as f:
                config = json.load(f)
                TILT_FEATURES = config['features']
        elif servo == 'pan':
            with open(directory + f'/weights/{servo}_{model_name_suffix}_config.json', 'r') as f:
                config = json.load(f)
                PAN_FEATURES = config['features']
        else:
            raise ValueError(f"Unknown servo: {servo}")

if not is_dual_output_mlp:
    print(f"Tilt features ({len(TILT_FEATURES)}): {TILT_FEATURES}", file=sys.stderr)
    print(f"Pan features ({len(PAN_FEATURES)}): {PAN_FEATURES}", file=sys.stderr)

# Global variable for shared memory map, to be cleaned up in signal_handler
shm_map = None
shm_handle = None

# Global buffers for temporal models (TCN/RNN/MLP)
tilt_input_buffer = []
pan_input_buffer = []
mlp_input_buffer = []  # For dual-output MLP models
BUFFER_SIZE = 10  # For TCN models

# Global storage for predicted current values to use as inputs for subsequent predictions
predicted_current_values = {'tilt': None, 'pan': None}
# Servo-specific flags for using predictions as input (reset at start of each transition)
use_prediction_as_input = {'tilt': False, 'pan': False}
prediction_count = 0  # Count of predictions made


#Use a averge of predicted and actual current as inut to the network
USE_BLENDING = True  # Set to False to use hard switching instead of blending
ACTUAL_CURRENT_WEIGHT = 0.20  # Weight for actual current in blending
PREDICTED_CURRENT_WEIGHT = 0.80  # Weight for predicted current in blending


def get_feature_value(input_data, feature, servo_name, first_prediction=True, use_predicted_current=False, predicted_currents=None):
    """
    Extracts the value of a specific feature from the input data dictionary.
    Handles both servo-specific features and common IMU features.
    
    Args:
        input_data: Input data dictionary
        feature: Feature name to extract
        servo_name: Current servo name being processed
        first_prediction: Whether this is the first prediction (deprecated, kept for compatibility)
        use_predicted_current: Whether to use predicted current values instead of actual
        predicted_currents: Dictionary containing predicted current values {'tilt': value, 'pan': value}
    """
    # Map from feature name to input_data['servo_data'] keys
    servo_feature_map = {
        'Position': 'position',
        'DistToGoal': 'goal_distance',
        'DistToStart': 'start_distance',
        'GoalPosition': 'goal_position',
        'StartPosition': 'starting_position',
        'Current': 'current'  # Add current mapping
    }

    # Handle IMU features
    if feature in ['GyroX', 'GyroY', 'GyroZ']:
        return input_data['gyro'][['GyroX', 'GyroY', 'GyroZ'].index(feature)]
    if feature in ['AccelX', 'AccelY', 'AccelZ']:
        return input_data['accel'][['AccelX', 'AccelY', 'AccelZ'].index(feature)]
    if feature in ['AngleX', 'AngleY', 'AngleZ']:
        return input_data['angles'][['AngleX', 'AngleY', 'AngleZ'].index(feature)]

    # --- BUG FIX ---
    # The original logic incorrectly looped through prefixes, which could cause a feature
    # for one servo (e.g., 'TiltPosition') to be incorrectly claimed by another servo's
    # model if not handled carefully.
    # The corrected logic determines the target servo and metric from the feature name
    # itself, ensuring the correct value is always retrieved.
    
    target_servo_prefix = None
    if feature.startswith('Tilt'):
        target_servo_prefix = 'Tilt'
        servo_idx = 0
    elif feature.startswith('Pan'):
        target_servo_prefix = 'Pan'
        servo_idx = 1
    else:
        # If the feature does not start with a known servo prefix and wasn't an IMU feature,
        # it's an unknown feature.
        raise ValueError(f"Unknown feature: {feature}")

    # Extract the metric from the feature name (e.g., 'TiltPosition' -> 'Position')
    metric = feature[len(target_servo_prefix):]
    
    if metric in servo_feature_map:
        data_key = servo_feature_map[metric]
        
        # Special handling for current values - use predicted values if available and requested
        if data_key == 'current' and use_predicted_current and predicted_currents is not None:
            target_servo_name = target_servo_prefix.lower()  # 'Tilt' -> 'tilt', 'Pan' -> 'pan'
            if target_servo_name in predicted_currents and predicted_currents[target_servo_name] is not None:
                # The predicted_currents are already in normalized space, so return them directly
            
                return predicted_currents[target_servo_name]
            else:
                print(f"No predicted current available for {target_servo_prefix}, using actual current", file=sys.stderr)
        
        return input_data['servo_data'][servo_idx][data_key]
    
    # If the metric is not in the map, the feature name is invalid.
    raise ValueError(f"Unknown feature metric '{metric}' in feature '{feature}'")

def parse_input_array_to_dict(flat_input_array, num_servos_expected):
    """
    Parses a flat numpy array of input data from shared memory into the
    dictionary structure expected by the `normalise_input` function.
    Assumes a fixed order: gyro(3), accel(3), angles(3),
    then for each servo: position(1), dist_to_goal(1), dist_to_start(1),
    goal_pos(1), start_pos(1).
    """
    ptr = 0
    gyro = flat_input_array[ptr:ptr+3]; ptr += 3
    accel = flat_input_array[ptr:ptr+3]; ptr += 3
    angles = flat_input_array[ptr:ptr+3]; ptr += 3

    servo_data_list = []
    positions = []
    goal_distances = []
    start_distances = []
    goal_positions = []
    starting_positions = []

   
    for i in range(num_servos_expected):
        pos = flat_input_array[ptr]; ptr += 1
        dist_goal = flat_input_array[ptr]; ptr += 1
        dist_start = flat_input_array[ptr]; ptr += 1
        goal_pos = flat_input_array[ptr]; ptr += 1
        start_pos = flat_input_array[ptr]; ptr += 1
        current_val = flat_input_array[ptr]; ptr += 1

        positions.append(pos)
        goal_distances.append(dist_goal)
        start_distances.append(dist_start)
        goal_positions.append(goal_pos)
        starting_positions.append(start_pos)
        
        servo_data_list.append({
            'position': pos,
            'goal_distance': dist_goal,
            'start_distance': dist_start,
            'goal_position': goal_pos,
            'starting_position': start_pos,
            'current': current_val
        })

    if ptr != len(flat_input_array):
        print(f"Warning: Parsed {ptr} input values, but received {len(flat_input_array)} from C++.", file=sys.stderr)

    # For debugging, print the parsed values similar to the old script
    # print("Parsed Gyro: ", gyro, file=sys.stderr)
    # print("Parsed Accel: ", accel, file=sys.stderr)
    # print("Parsed Angles: ", angles, file=sys.stderr)
    # print("Parsed Positions: ", positions, file=sys.stderr)
    # print("Parsed Goal distances: ", goal_distances, file=sys.stderr)
    # print("Parsed Start distances: ", start_distances, file=sys.stderr)
    # print("Parsed Goal positions: ", goal_positions, file=sys.stderr)
    # print("Parsed Starting positions: ", starting_positions, file=sys.stderr)

    return {
        'gyro': gyro,
        'accel': accel,
        'angles': angles,
        'servo_data': servo_data_list,
        'num_servos': num_servos_expected
        
    }

def get_flag(shm_map_obj, offset):
    """Reads a boolean flag from shared memory."""
    flag_byte = struct.unpack_from('<?', shm_map_obj, offset)[0]
    return bool(flag_byte)

def set_flag(shm_map_obj, offset, value):
    """Writes a boolean flag to shared memory."""
    shm_map_obj.seek(offset) # Important: mmap object needs seek before write if not sequential
    shm_map_obj.write(struct.pack('<?', bool(value)))
    shm_map_obj.flush() # Ensure it's written

def normalise_input(input_data, means_stds, servo_name, model_name, feature_list, scalar_object=None, use_predicted_current=False, predicted_currents=None):
    """
    Normalise input data for a specific servo model using a custom feature list.
    For temporal models (RNN/TCN) and dual-output MLP models, manages input buffers and only returns data when enough timesteps are collected.
    
    Args:
        input_data: Input data dictionary
        means_stds: Normalization statistics
        servo_name: Name of the servo
        model_name: Name of the model
        feature_list: List of features to extract
        scalar_object: Optional StandardScaler object
        use_predicted_current: Whether to use predicted current values instead of actual
        predicted_currents: Dictionary containing predicted current values
    """
    global tilt_input_buffer, pan_input_buffer, mlp_input_buffer
    
    input_array = []
    for feature in feature_list:
        value = get_feature_value(input_data, feature, servo_name, first_prediction=True, 
                                use_predicted_current=use_predicted_current, 
                                predicted_currents=predicted_currents)
        input_array.append(value)
    
    x = np.array(input_array, dtype=np.float32)

    # Choose normalization method
    if scalar_object is not None:
        # Use StandardScaler object (preferred for RNN models)
        x_df = pd.DataFrame([dict(zip(feature_list, x.flatten()))], dtype=np.float32)
        expected_feature_order = scalar_object.feature_names_in_
        if not set(feature_list) == set(expected_feature_order):
            raise ValueError(f"Feature mismatch for servo '{servo_name}'. Model expects {set(feature_list)} but scaler was trained on {set(expected_feature_order)}.")
        
        x_df_ordered = x_df[expected_feature_order]
        x = scalar_object.transform(x_df_ordered).astype(np.float32)
        x = x.reshape(1, -1)
    else:
        # Use mean/std from JSON (equivalent to StandardScaler)
        means = np.array([means_stds[feature]['mean'] for feature in feature_list], dtype=np.float32)
        stds = np.array([means_stds[feature]['std'] for feature in feature_list], dtype=np.float32)
        
        if 'raw' in model_name:
            x = x.reshape(1, -1)
        else:
            # Normalize each feature, but skip normalization for predicted current values (they're already normalized)
            x_normalized = np.zeros_like(x)
            for i, (feature, value) in enumerate(zip(feature_list, x)):
                if (use_predicted_current and feature.endswith('Current') and 
                    predicted_currents is not None and 
                    ((feature.startswith('Tilt') and predicted_currents.get('tilt') is not None) or
                     (feature.startswith('Pan') and predicted_currents.get('pan') is not None))):
                    # This is a predicted current value, already normalized - use as is
                    x_normalized[i] = value
                    print(f"Skipping normalization for predicted current {feature}: {value:.4f}", file=sys.stderr)
                else:
                    # Regular feature - apply normalization
                    x_normalized[i] = (value - means[i]) / stds[i]
            
            x = x_normalized.reshape(1, -1).astype(np.float32)

    # Handle RNN and TCN models - use real temporal data with buffers
    if 'rnn' in model_name.lower():
        timesteps_match = re.search(r'TimeSteps(\d+)', model_name)
        timesteps = int(timesteps_match.group(1)) if timesteps_match else 5
        
        # Add current input to buffer
        if servo_name == 'tilt':
            tilt_input_buffer.append(x[0])  # x[0] to get the 1D array
            if len(tilt_input_buffer) > timesteps:
                tilt_input_buffer.pop(0)  # Remove oldest
            
            # Only return data when we have enough timesteps
            if len(tilt_input_buffer) < timesteps:
                return None  # Not enough data yet
            
            # Stack the buffer into the required shape
            x_rnn = np.array(tilt_input_buffer).reshape(1, timesteps, -1)
        else:  # pan
            pan_input_buffer.append(x[0])
            if len(pan_input_buffer) > timesteps:
                pan_input_buffer.pop(0)
            
            if len(pan_input_buffer) < timesteps:
                return None
            
            x_rnn = np.array(pan_input_buffer).reshape(1, timesteps, -1)
        
        return x_rnn.astype(np.float32)
        
    elif 'temporal_mlp' in model_name.lower():
        timesteps_match = re.search(r'TimeSteps(\d+)', model_name)
        timesteps = int(timesteps_match.group(1)) if timesteps_match else 1  # Default to 1 if not specified
        
        
        # Add current input to buffer
        if servo_name == 'tilt':
            tilt_input_buffer.append(x[0])  # x[0] to get the 1D array
            if len(tilt_input_buffer) > timesteps:
                tilt_input_buffer.pop(0)  # Remove oldest
            
            # Only return data when we have enough timesteps
            if len(tilt_input_buffer) < timesteps:
                return None  # Not enough data yet
            
            # Stack the buffer into the required shape for temporal MLP
            x_temporal = np.array(tilt_input_buffer).reshape(1, timesteps, -1)
        else:  # pan
            pan_input_buffer.append(x[0])
            if len(pan_input_buffer) > timesteps:
                pan_input_buffer.pop(0)
            
            if len(pan_input_buffer) < timesteps:
                return None
            
            x_temporal = np.array(pan_input_buffer).reshape(1, timesteps, -1)
        
        return x_temporal.astype(np.float32)
        
    elif 'tcn' in model_name.lower():
        # TCN models use 10 timesteps
        timesteps = 10
        
        # Add current input to buffer
        if servo_name == 'tilt':
            tilt_input_buffer.append(x[0])  # x[0] to get the 1D array
            if len(tilt_input_buffer) > timesteps:
                tilt_input_buffer.pop(0)  # Remove oldest
            
            # Only return data when we have enough timesteps
            if len(tilt_input_buffer) < timesteps:
                return None  # Not enough data yet
            
            # Stack the buffer into the required shape
            x_tcn = np.array(tilt_input_buffer).reshape(1, timesteps, -1)
        else:  # pan
            pan_input_buffer.append(x[0])
            if len(pan_input_buffer) > timesteps:
                pan_input_buffer.pop(0)
            
            if len(pan_input_buffer) < timesteps:
                return None
            
            x_tcn = np.array(pan_input_buffer).reshape(1, timesteps, -1)
        
        return x_tcn.astype(np.float32)
    
    return x


def convert_to_tflite_if_possible(keras_model, weights_path, model_name, is_temporal_mlp=False, is_tcn=False):
    """
    Helper function to convert Keras models to TFLite with proper validation.
    """
    try:
        # Generate proper TFLite path
        if weights_path.endswith('.keras'):
            tflite_model_path = weights_path.replace('.keras', '.tflite')
        elif weights_path.endswith('.weights.h5'):
            tflite_model_path = weights_path.replace('.weights.h5', '.tflite')
        else:
            tflite_model_path = weights_path + '.tflite'
        
        print(f"TFLite path: {tflite_model_path}, exists: {os.path.exists(tflite_model_path)}", file=sys.stderr)
        
        # Check if TFLite model already exists
        if os.path.exists(tflite_model_path):
            try:
                print(f"Loading existing TFLite model: {tflite_model_path}", file=sys.stderr)
                interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
                interpreter.allocate_tensors()
                
                # Validate shapes
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                input_shape = input_details[0]['shape']
                output_shape = output_details[0]['shape']
                
                if is_temporal_mlp:
                    # Temporal MLP expects 3D input (batch, time_steps, features)
                    if len(input_shape) != 3:
                        print(f"Warning: Expected 3D input for temporal MLP, got {input_shape}", file=sys.stderr)
                elif is_tcn:
                    # TCN also expects 3D input
                    if len(input_shape) != 3 or input_shape[1] != 10:
                        print(f"Warning: Expected (1, 10, features) for TCN, got {input_shape}", file=sys.stderr)
                
                print(f"TFLite model shapes - Input: {input_shape}, Output: {output_shape}", file=sys.stderr)
                return interpreter
            except Exception as load_e:
                print(f"Failed to load existing TFLite model {tflite_model_path}: {load_e}. Will convert from Keras.", file=sys.stderr)
                # Continue to conversion below
        
        # Convert Keras model to TFLite
        print(f"Converting {'temporal MLP' if is_temporal_mlp else 'TCN' if is_tcn else 'standard'} model to TFLite...", file=sys.stderr)
        
        # Log Keras model details for debugging
        keras_input_shape = keras_model.input.shape if hasattr(keras_model, 'input') else 'Unknown'
        keras_output_shape = keras_model.output.shape if hasattr(keras_model, 'output') else 'Unknown'
        print(f"Keras model shapes - Input: {keras_input_shape}, Output: {keras_output_shape}", file=sys.stderr)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        
        # For temporal models, we might need specific optimizations
        if is_temporal_mlp or is_tcn:
            # Add any specific optimizations for temporal models if needed
            pass
        
        tflite_model_content = converter.convert()
        
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model_content)
        
        print(f"Successfully converted to TFLite: {tflite_model_path}", file=sys.stderr)
        
        # Load and validate the converted model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # Test with dummy input
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        dummy_input = np.zeros(input_details[0]['shape'], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        dummy_output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"TFLite conversion validation successful - output shape: {dummy_output.shape}", file=sys.stderr)
        return interpreter
        
    except Exception as e:
        print(f"TFLite conversion failed: {e}. Falling back to Keras model.", file=sys.stderr)
        return keras_model


def create_model_with_weights(weights_path, model_name, num_inputs):
    # Importing json here as it's only used in this function and main
    import json 
    
    # Check for .keras file first (much simpler and faster loading)
    keras_file_path = weights_path.replace('.weights.h5', '.keras')
    if os.path.exists(keras_file_path):
        try:
            print(f"Loading .keras model from {keras_file_path}", file=sys.stderr)
            keras_model = load_model(keras_file_path, compile=False)  # Don't compile for inference
            keras_model.trainable = False  # Optimize for inference
            print(f"Successfully loaded .keras model: {keras_file_path}", file=sys.stderr)
            
            # Check if this is a temporal MLP model (trained with time steps)
            if 'temporal_mlp' in model_name.lower():
                print(f"Detected temporal MLP model - will convert to TFLite", file=sys.stderr)
                # Temporal MLP models use Dense layers with Flatten, so they can be converted to TFLite
                return convert_to_tflite_if_possible(keras_model, keras_file_path, model_name, is_temporal_mlp=True)
            elif 'rnn' in model_name.lower() or 'lstm' in model_name.lower() or 'gru' in model_name.lower():
                print(f"RNN model detected in .keras file - using Keras model directly", file=sys.stderr)
                return keras_model
            elif 'tcn' in model_name.lower():
                print(f"TCN model detected - will convert to TFLite", file=sys.stderr)
                return convert_to_tflite_if_possible(keras_model, keras_file_path, model_name, is_tcn=True)
            else:
                print(f"Standard model detected - will convert to TFLite", file=sys.stderr)
                return convert_to_tflite_if_possible(keras_model, keras_file_path, model_name)
            
            return keras_model
        except Exception as e:
            print(f"Error loading .keras model {keras_file_path}: {e}. Falling back to weights loading.", file=sys.stderr)
    
    tflite_model_path = weights_path.replace('.weights.h5', '.tflite')

    # Check if this is an RNN or TCN model early - RNN models can't use TFLite, but TCN can
    is_rnn_model = 'rnn' in model_name.lower() or 'lstm' in model_name.lower() or 'gru' in model_name.lower()
    is_tcn_model = 'tcn' in model_name.lower()
    is_sequential_model = is_rnn_model or is_tcn_model
    
    # Try to load TFLite model if it exists and is not an RNN model (TCN models can use TFLite)
    if os.path.exists(tflite_model_path) and not is_rnn_model:
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            interpreter.allocate_tensors()
            print(f"Successfully loaded TFLite model", file=sys.stderr)
            # You might want to verify input/output tensor details here if needed
            return interpreter
        except Exception as e:
            print(f"Error loading existing TFLite model {tflite_model_path}: {e}. Will try to convert Keras model.", file=sys.stderr)

    # If TFLite model doesn't exist or failed to load, load Keras model and convert
    keras_model = None
    config_path = weights_path.replace('.weights.h5', '_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            keras_model = Sequential()
            keras_model.name = model_name
            features = list(config['features'])
            
            # Check if this is an RNN model by looking for LSTM or GRU layers in config
            config_is_rnn_model = any(layer['class_name'] in ['LSTM', 'GRU'] for layer in config['layers'])
            is_rnn_model = is_rnn_model or config_is_rnn_model  # Update our flag
            
            if is_rnn_model:
                # For RNN models, we need to determine the timesteps
                # From the model name, extract timesteps (e.g., "TimeSteps5" -> 5)
                timesteps_match = re.search(r'TimeSteps(\d+)', model_name)
                if timesteps_match:
                    timesteps = int(timesteps_match.group(1))
                else:
                    timesteps = 5  # Default fallback
                keras_model.add(Input(shape=(timesteps, len(features))))
            elif is_tcn_model:
                # For TCN models, use 10 timesteps based on your architecture
                timesteps = 10
                keras_model.add(Input(shape=(timesteps, len(features))))
            else:
                keras_model.add(Input(shape=(len(features),)))
            
            for layer_config_item in config['layers']: # Renamed to avoid conflict
                if layer_config_item['class_name'] == 'Dense':
                    dense_config = layer_config_item['config'] # Renamed to avoid conflict
                    regularizer = None
                    if dense_config.get('kernel_regularizer'):
                        reg_config = dense_config['kernel_regularizer']['config']
                        regularizer = l2(reg_config.get('l2', 0))
                    
                    keras_model.add(Dense(
                        units=dense_config['units'],
                        activation=dense_config['activation'],
                        kernel_regularizer=regularizer
                    ))
                elif layer_config_item['class_name'] == 'LSTM':
                    lstm_config = layer_config_item['config']
                    regularizer = None
                    if lstm_config.get('kernel_regularizer'):
                        reg_config = lstm_config['kernel_regularizer']['config']
                        regularizer = l2(reg_config.get('l2', 0))
                    
                    keras_model.add(LSTM(
                        units=lstm_config['units'],
                        activation=lstm_config.get('activation', 'tanh'),
                        recurrent_activation=lstm_config.get('recurrent_activation', 'sigmoid'),
                        return_sequences=lstm_config.get('return_sequences', False),
                        return_state=lstm_config.get('return_state', False),
                        go_backwards=lstm_config.get('go_backwards', False),
                        stateful=lstm_config.get('stateful', False),
                        unroll=lstm_config.get('unroll', False),
                        use_bias=lstm_config.get('use_bias', True),
                        kernel_regularizer=regularizer,
                        dropout=lstm_config.get('dropout', 0.0),
                        recurrent_dropout=lstm_config.get('recurrent_dropout', 0.0)
                    ))
                elif layer_config_item['class_name'] == 'GRU':
                    gru_config = layer_config_item['config']
                    regularizer = None
                    if gru_config.get('kernel_regularizer'):
                        reg_config = gru_config['kernel_regularizer']['config']
                        regularizer = l2(reg_config.get('l2', 0))
                    
                    keras_model.add(GRU(
                        units=gru_config['units'],
                        activation=gru_config.get('activation', 'tanh'),
                        recurrent_activation=gru_config.get('recurrent_activation', 'sigmoid'),
                        return_sequences=gru_config.get('return_sequences', False),
                        return_state=gru_config.get('return_state', False),
                        go_backwards=gru_config.get('go_backwards', False),
                        stateful=gru_config.get('stateful', False),
                        unroll=gru_config.get('unroll', False),
                        use_bias=gru_config.get('use_bias', True),
                        kernel_regularizer=regularizer,
                        dropout=gru_config.get('dropout', 0.0),
                        recurrent_dropout=gru_config.get('recurrent_dropout', 0.0)
                    ))
                elif layer_config_item['class_name'] == 'Activation':
                    activation_config = layer_config_item['config']
                    keras_model.add(Activation(activation_config['activation']))
                elif layer_config_item['class_name'] == 'BatchNormalization':
                    keras_model.add(BatchNormalization())
                elif layer_config_item['class_name'] == 'Dropout':
                    keras_model.add(Dropout(layer_config_item['config']['rate']))
            
            # Don't compile for inference-only usage (faster loading)
            # keras_model.compile(optimizer=Adam(learning_rate=0.001), 
            #                   loss='mean_squared_error', 
            #                   metrics=['mae'])
            
            keras_model.load_weights(weights_path)
            
            # Optimize for inference
            keras_model.trainable = False  # Disable training mode
            
            print(f"Successfully loaded Keras model from config: {config_path}", file=sys.stderr)
            
        except Exception as e:
            print(f"Error loading Keras model from config: {e}", file=sys.stderr)
            keras_model = None # Reset to ensure fallback is used if config loading fails partially

    if keras_model is None: # Fallback to default Keras model architecture if config failed or no weights_path
        print(f"Using default Keras model architecture for {model_name} (or weights_path was None)", file=sys.stderr)
        keras_model = Sequential()
        keras_model.name = model_name
        keras_model.add(Input(shape=(num_inputs,)))  
        keras_model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
        keras_model.add(BatchNormalization())
        keras_model.add(Dropout(0.3))
        keras_model.add(Dense(32, activation='relu'))
        keras_model.add(BatchNormalization())
        keras_model.add(Dropout(0.3))
        keras_model.add(Dense(1, activation='linear'))
        keras_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
        if weights_path and os.path.exists(weights_path):
            keras_model.load_weights(weights_path)
        else:
            print(f"Warning: No Keras weights file found at {weights_path}. Model will be uninitialized.", file=sys.stderr)

    # For RNN models, return Keras model directly without TFLite conversion
    # TCN models can be converted to TFLite since they use standard Conv1D layers
    if is_rnn_model:
        print(f"RNN model detected - using Keras model directly (skipping TFLite conversion)", file=sys.stderr)
        return keras_model
    
    # Convert Keras model to TFLite (works for regular Dense models and TCN models)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        
        # For TCN models with 3D inputs, we might need specific optimizations
        if is_tcn_model:
            print(f"Converting TCN model to TFLite...", file=sys.stderr)
            # TCN models should work with default settings, but we can add optimizations if needed
            # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # You can enable optimizations like float16 quantization here if desired
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
        tflite_model_content = converter.convert()
        
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model_content)
        
        model_type = "TCN" if is_tcn_model else "Dense"
        print(f"Successfully converted {model_type} Keras model to TFLite and saved to {tflite_model_path}", file=sys.stderr)
        
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path) # Load from file to be consistent
        # Or use: interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
        interpreter.allocate_tensors()
        
        # Verify the input/output shapes for TCN models
        if is_tcn_model:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_shape = input_details[0]['shape']
            output_shape = output_details[0]['shape']
            
            print(f"TFLite TCN model - Input shape: {input_shape}, Output shape: {output_shape}", file=sys.stderr)
            
            # Validate shapes for TCN models
            if len(input_shape) != 3:
                print(f"Warning: TCN model input should be 3D (batch, timesteps, features), got {len(input_shape)}D", file=sys.stderr)
            if input_shape[1] != 10:
                print(f"Warning: TCN model expects 10 timesteps, got {input_shape[1]}", file=sys.stderr)
            if len(output_shape) != 2 or output_shape[1] != 1:
                print(f"Warning: TCN model output should be (batch, 1), got {output_shape}", file=sys.stderr)
            
            # Test with a dummy input to verify the conversion worked
            try:
                dummy_input = np.zeros(input_shape, dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], dummy_input)
                interpreter.invoke()
                dummy_output = interpreter.get_tensor(output_details[0]['index'])
                print(f"TCN TFLite model validation successful - output shape: {dummy_output.shape}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: TCN TFLite model validation failed: {e}", file=sys.stderr)
        
        return interpreter
    except Exception as e:
        print(f"Error converting Keras model to TFLite: {e}", file=sys.stderr)
        if is_tcn_model:
            print(f"TCN model TFLite conversion failed - falling back to Keras model", file=sys.stderr)
            return keras_model
        else:
            raise # Re-raise the exception for non-TCN models as we can't proceed without a model


def signal_handler(sig, frame):
    global shm_map, shm_handle
    print("Received termination signal. Cleaning up...", file=sys.stderr)
    if shm_map is not None:
        try:
            # Signal C++ that we are shutting down if possible (though C++ usually drives shutdown)
            # This script mostly reacts to C++'s shutdown_signal.
            # If we wanted to proactively tell C++ we are dying unexpectedly,
            # we'd need a separate flag or mechanism.
            # For now, just ensure flags are in a known state if an error occurs before setting python_wrote_output.
            # If C++ is waiting on python_wrote_output, it might timeout.
            # The primary role here is to clean up Python-side resources.
            print("Closing mmap object.", file=sys.stderr)
            shm_map.close()
            shm_map = None
        except Exception as e:
            print(f"Error closing mmap object during cleanup: {e}", file=sys.stderr)
            
    if shm_handle is not None:
        try:
            print("Closing shared memory file descriptor.", file=sys.stderr)
            shm_handle.close_fd()
            shm_handle = None
        except Exception as e:
            print(f"Error closing shared memory FD during cleanup: {e}", file=sys.stderr)
    sys.exit(0)

def main():
    global shm_map, shm_handle # Allow signal_handler to access these
    import json # Import json here as it's used in main

    try:
        # Register signal handlers for proper cleanup
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # --- Argument Parsing ---
        if len(sys.argv) != 5:
            print("Usage: python_script.py <shm_base_name> <num_inputs_cpp> <max_outputs_cpp> <flags_struct_size>", file=sys.stderr)
            sys.exit(1)

        shm_base_name = sys.argv[1]
        try:
            num_inputs_cpp = int(sys.argv[2])
            max_outputs_cpp = int(sys.argv[3]) # Max number of float data elements Python can write
            flags_struct_size = int(sys.argv[4]) # Size of the flag structure in bytes
        except ValueError:
            print("Error: num_inputs, max_outputs, and flags_struct_size must be integers.", file=sys.stderr)
            sys.exit(1)

        shm_name = "/" + shm_base_name
        print(f"Python script starting with SHM name: {shm_name}, Inputs: {num_inputs_cpp}, Max Outputs: {max_outputs_cpp}, Flags Size: {flags_struct_size}", file=sys.stderr)

        expected_total_inputs = get_expected_total_inputs()
        if num_inputs_cpp != expected_total_inputs:
            print(f"Error: Received {num_inputs_cpp} inputs from C++, but script expects {expected_total_inputs} for {NUM_SERVOS} servos.", file=sys.stderr)
            sys.exit(1)
        
        # Max outputs this script will produce is NUM_SERVOS (one per servo current)
        # Ensure C++ is configured to expect at least this many.
        if max_outputs_cpp < NUM_SERVOS:
            print(f"Error: C++ expects max {max_outputs_cpp} outputs, but script produces {NUM_SERVOS}. Please configure 'NumberOutputs' in Ikaros for PythonScriptCaller >= {NUM_SERVOS}.", file=sys.stderr)
            sys.exit(1)

        # --- Shared Memory Setup ---
        # Calculate total size of the data portion (inputs + 1_for_output_count + outputs)
        input_data_size = num_inputs_cpp * FLOAT_SIZE
        # Python writes: 1 float for actual_output_count, then actual_output_count floats for data.
        # C++ allocates space for max_outputs_cpp data elements + 1 count.
        output_data_buffer_size = (1 + max_outputs_cpp) * FLOAT_SIZE
        
        total_shm_size = flags_struct_size + input_data_size + output_data_buffer_size
        
        # Define data offsets
        input_data_offset = flags_struct_size
        output_data_offset = flags_struct_size + input_data_size # Start of [count, data1, data2...]

        print(f"Calculated SHM: Total Size={total_shm_size}, InputDataOffset={input_data_offset}, OutputDataOffset={output_data_offset}", file=sys.stderr)

        try:
            shm_handle = posix_ipc.SharedMemory(shm_name)
            shm_map = mmap.mmap(shm_handle.fd, total_shm_size)
            # shm_handle.close_fd() # According to docs, fd can be closed after mmap
            print(f"Successfully mapped shared memory '{shm_name}' of size {total_shm_size} bytes.", file=sys.stderr)
        except posix_ipc.ExistentialError:
            print(f"Error: Shared memory segment '{shm_name}' does not exist. Was it created by C++?", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing shared memory: {e}", file=sys.stderr)
            sys.exit(1)
        
        # --- Model Loading ---
        
        models = {}
        means_stds = {}
        num_features_for_model_init = 0 # To store feature count for dummy input

        if is_dual_output_mlp:
            # For dual-output MLP models, load a single model that predicts both servos
            num_inputs_for_this_model = len(ALL_FEATURES)
            weights_path = directory + f'/weights/{model_name_suffix}.weights.h5'
            
            # Load the single model for both servos
            models['dual_mlp'] = create_model_with_weights(weights_path, model_name_suffix, num_inputs_for_this_model)
            
            # Load mean/std data for normalization (single file for all features)
            mean_std_path = directory + f'/weights/{mean_stds_suffix}'
            if os.path.exists(mean_std_path):
                with open(mean_std_path) as f:
                    means_stds['dual_mlp'] = json.load(f)
                print(f"Loaded dual MLP mean/std: {mean_std_path}", file=sys.stderr)
            else:
                print(f"Warning: Dual MLP mean/std file not found: {mean_std_path}. Using default mean/std file.", file=sys.stderr)
                # Fallback to default suffix file
                with open(directory + f'/weights/{mean_stds_suffix}') as f:
                    means_stds['dual_mlp'] = json.load(f)
                    
            print(f"Loaded dual-output MLP model with {num_inputs_for_this_model} features", file=sys.stderr)
        else:
            # Original behavior for separate models per servo
            for servo in servos:
                feature_list_for_servo = TILT_FEATURES if servo == 'tilt' else PAN_FEATURES
                num_inputs_for_this_model = len(feature_list_for_servo)
           
                weights_path = directory + f'/weights/{servo}_{model_name_suffix}.weights.h5'
                models[servo] = create_model_with_weights(weights_path, model_name_suffix, num_inputs_for_this_model)
                
                # Load mean/std data for normalization
                if 'temporal_mlp' in model_name_suffix.lower():
                    # For temporal MLP models, use the specific mean/std files generated during training
                    mean_std_path = directory + f'/weights/{servo}_temporal_mlp_mean_std.json'
                    if os.path.exists(mean_std_path):
                        with open(mean_std_path) as f:
                            means_stds[servo] = json.load(f)
                        print(f"Loaded temporal MLP mean/std for {servo}: {mean_std_path}", file=sys.stderr)
                    else:
                        print(f"Warning: Temporal MLP mean/std file not found: {mean_std_path}. Using default mean/std file.", file=sys.stderr)
                        with open(directory + f'/weights/{servo}_{mean_stds_suffix}') as f:
                            means_stds[servo] = json.load(f)
                else:
                    with open(directory + f'/weights/{servo}_{mean_stds_suffix}') as f:
                        means_stds[servo] = json.load(f)
        
        # Model warm-up 
        if is_dual_output_mlp:
            # Warm up the dual-output MLP model
            num_inputs_for_this_model = len(ALL_FEATURES)
            if num_inputs_for_this_model > 0:
                # Create dummy input with correct shape for model type
                if 'mlp' in model_name_suffix.lower() and re.search(r'TimeSteps(\d+)', model_name_suffix):
                    # MLP models with timesteps need 3D input: (batch_size, timesteps, features)
                    timesteps_match = re.search(r'TimeSteps(\d+)', model_name_suffix)
                    timesteps = int(timesteps_match.group(1)) if timesteps_match else 5
                    dummy_input = np.zeros((1, timesteps, num_inputs_for_this_model), dtype=np.float32)
                else:
                    # Regular MLP models need 2D input: (batch_size, features)
                    dummy_input = np.zeros((1, num_inputs_for_this_model), dtype=np.float32)
                
                model = models['dual_mlp']
                
                # Check if it's a TFLite interpreter or Keras model
                if hasattr(model, 'get_input_details'):  # TFLite interpreter
                    input_details = model.get_input_details()
                    model.set_tensor(input_details[0]['index'], dummy_input.astype(np.float32))
                    model.invoke()
                    _ = model.get_tensor(model.get_output_details()[0]['index'])
                else:  # Keras model
                    # Warm up with multiple predictions to optimize internal caching
                    for _ in range(3):
                        _ = model(dummy_input, training=False)
        
                print(f"Dual-output MLP model warmed up with input shape {dummy_input.shape}.", file=sys.stderr)
                
                # Print actual model input shape for debugging
                if hasattr(model, 'input_shape'):
                    print(f"Model's expected input_shape: {model.input_shape}", file=sys.stderr)
                elif hasattr(model, 'get_input_details'):
                    input_details = model.get_input_details()
                    print(f"TFLite model's expected input shape: {input_details[0]['shape']}", file=sys.stderr)
            else:
                print("Warning: No features found for dual MLP model warm-up. Skipping.", file=sys.stderr)
        else:
            # Original behavior for separate models
            for servo in servos:
                feature_list_for_servo = TILT_FEATURES if servo == 'tilt' else PAN_FEATURES
                num_inputs_for_this_model = len(feature_list_for_servo)
                
                if num_inputs_for_this_model > 0:
                    # Create dummy input with correct shape for model type
                    if 'rnn' in model_name_suffix.lower():
                        # RNN models need 3D input: (batch_size, timesteps, features)
                        timesteps_match = re.search(r'TimeSteps(\d+)', model_name_suffix)
                        timesteps = int(timesteps_match.group(1)) if timesteps_match else 5
                        dummy_input = np.zeros((1, timesteps, num_inputs_for_this_model), dtype=np.float32)
                    elif 'temporal_mlp' in model_name_suffix.lower():
                        # Temporal MLP models need 3D input: (batch_size, timesteps, features)
                        timesteps_match = re.search(r'TimeSteps(\d+)', model_name_suffix)
                        timesteps = int(timesteps_match.group(1)) if timesteps_match else 10  # Default to 10 based on your training
                        dummy_input = np.zeros((1, timesteps, num_inputs_for_this_model), dtype=np.float32)
                    elif 'tcn' in model_name_suffix.lower():
                        # TCN models need 3D input: (batch_size, timesteps, features)
                        timesteps = 10  # Based on your model architecture
                        dummy_input = np.zeros((1, timesteps, num_inputs_for_this_model), dtype=np.float32)
                    else:
                        # Regular models need 2D input: (batch_size, features)
                        dummy_input = np.zeros((1, num_inputs_for_this_model), dtype=np.float32)
                    
                    model = models[servo]
                    
                    # Check if it's a TFLite interpreter or Keras model
                    if hasattr(model, 'get_input_details'):  # TFLite interpreter
                        input_details = model.get_input_details()
                        model.set_tensor(input_details[0]['index'], dummy_input.astype(np.float32))
                        model.invoke()
                        _ = model.get_tensor(model.get_output_details()[0]['index'])
                    else:  # Keras model
                        # Warm up with multiple predictions to optimize internal caching
                        for _ in range(3):
                            _ = model(dummy_input, training=False)
            
                    print(f"Model '{servo}' warmed up with input shape {dummy_input.shape}.", file=sys.stderr)
                else:
                    print("Warning: No features found for model warm-up. Skipping.", file=sys.stderr)

        print(f"Python script '{shm_name}' initialized and waiting for data.", file=sys.stderr)

        # --- Main Loop ---
        while True:
            if get_flag(shm_map, SHUTDOWN_SIGNAL_OFFSET):
                print("Shutdown signal received from C++. Exiting ANN prediction loop.", file=sys.stderr)
                break

            if get_flag(shm_map, CPP_WROTE_INPUT_OFFSET):
                t_start_read = time.perf_counter()
                
                # Read input data array
                shm_map.seek(input_data_offset)
                input_bytes = shm_map.read(input_data_size)
                flat_input_array = np.frombuffer(input_bytes, dtype=np.float32)
                
                t_end_read = time.perf_counter()

                # Parse flat array into dictionary
                input_data_dict = parse_input_array_to_dict(flat_input_array, NUM_SERVOS)
                
                predictions = []
                previous_predictions = []
               
                try:
                    if is_dual_output_mlp:
                        # For dual-output MLP models, make a single prediction for both servos
                        # Extract all features at once with validation
                        
                        global predicted_current_values, use_prediction_as_input, prediction_count
                        
                        # Determine if we should use predicted current values
                        has_current_features = 'current' in model_name_suffix.lower()
                        
                        # Extract features manually with proper normalization handling
                        input_array = []
                        feature_debug = []
                        
                        # Get normalization stats
                        means = np.array([means_stds['dual_mlp'][feature]['mean'] for feature in ALL_FEATURES], dtype=np.float32)
                        stds = np.array([means_stds['dual_mlp'][feature]['std'] for feature in ALL_FEATURES], dtype=np.float32)
                        
                        # VALIDATION: Verify all features have normalization stats
                        missing_features = []
                        for feature in ALL_FEATURES:
                            if feature not in means_stds['dual_mlp']:
                                missing_features.append(feature)
                        
                        if missing_features:
                            print(f"ERROR: Missing normalization stats for features: {missing_features}", file=sys.stderr)
                            print(f"Available keys: {list(means_stds['dual_mlp'].keys())}", file=sys.stderr)
                            set_flag(shm_map, CPP_WROTE_INPUT_OFFSET, False)
                            continue
                        
                        # Check for zero stds (would cause division by zero)
                        zero_std_features = [ALL_FEATURES[i] for i, std in enumerate(stds) if std == 0]
                        if zero_std_features:
                            print(f"WARNING: Features with zero std (will not normalize): {zero_std_features}", file=sys.stderr)
                        
                        for i, feature in enumerate(ALL_FEATURES):
                            # Determine which servo this feature belongs to based on prefix
                            if feature.startswith('Tilt'):
                                servo_for_feature = 'tilt'
                            elif feature.startswith('Pan'):
                                servo_for_feature = 'pan'
                            else:
                                # Handle IMU features that don't have servo prefix (Gyro, Accel, Angle)
                                servo_for_feature = 'tilt'  # Use tilt as default for IMU features
                            
                            # For current features, handle predicted vs actual values with blending
                            if feature.endswith('Current') and has_current_features:
                                # Determine which servo's current this is
                                target_servo_name = feature[:4].lower()  # 'TiltCurrent' -> 'tilt', 'PanCurrent' -> 'pan'
                                
                                # Get actual current value and normalize it
                                raw_actual = get_feature_value(input_data_dict, feature, servo_for_feature)
                                normalized_actual = (raw_actual - means[i]) / (stds[i] + 1e-8)
                                
                                # Check if we have a predicted value available
                                if (target_servo_name in predicted_current_values and 
                                    predicted_current_values[target_servo_name] is not None):
                                    # Get the prediction weight based on distance from start
                                    servo_idx = 0 if target_servo_name == 'tilt' else 1
                                    dist_to_start = input_data_dict['servo_data'][servo_idx]['start_distance']
                                    
                                    
                                    if USE_BLENDING:
                                        # Blend predicted and actual values
                                        normalized_predicted = predicted_current_values[target_servo_name]
                                        feature_value = ACTUAL_CURRENT_WEIGHT * normalized_actual + normalized_predicted* PREDICTED_CURRENT_WEIGHT
                                        

                                    else:
                                        # Weight is 0, use actual current only
                                        feature_value = normalized_actual
                                else:
                                    # No predicted value available yet, use actual current
                                    feature_value = normalized_actual
                                    if feature == 'TiltCurrent' and prediction_count < 5:
                                        print(f"Prediction #{prediction_count}: No predicted current yet, using actual for {target_servo_name}", 
                                              file=sys.stderr)
                            else:
                                # Regular feature - get raw value and normalize
                                raw_value = get_feature_value(input_data_dict, feature, servo_for_feature)
                                feature_value = (raw_value - means[i]) / (stds[i] + 1e-8)
                            
                            input_array.append(feature_value)
                            feature_debug.append((feature, feature_value))
                        
                        x_normalized = np.array(input_array, dtype=np.float32)
                        
                        # Handle temporal buffering for MLP models with timesteps
                        timesteps_match = re.search(r'TimeSteps(\d+)', model_name_suffix)
                        timesteps = int(timesteps_match.group(1)) if timesteps_match else 1
                        
                        # Add to buffer
                        mlp_input_buffer.append(x_normalized)
                        if len(mlp_input_buffer) > timesteps:
                            mlp_input_buffer.pop(0)
                        
                        # Check if we have enough timesteps
                        if len(mlp_input_buffer) < timesteps:
                            if len(mlp_input_buffer) == 1:  # Only print once
                                print(f"Initializing dual MLP buffer (needs {timesteps} timesteps)", file=sys.stderr)
                            set_flag(shm_map, CPP_WROTE_INPUT_OFFSET, False)
                            continue
                        
                        # Prepare input for model
                        # All temporal models (timesteps > 0) use 3D input: (batch, timesteps, features)
                        # This matches how create_sequences() works in training, even for timesteps=1
                        if timesteps > 0:
                            # Use only the required timesteps
                            model_input = np.array(mlp_input_buffer[-timesteps:]).reshape(1, timesteps, -1)
                        else:
                            # timesteps=0 means non-temporal model (2D input)
                            model_input = x_normalized.reshape(1, -1)
                        
                        # VALIDATION 4: Check input shape
                        # Remove this validation since it's causing issues - let the model validate itself
                        # The model's Input layer will check the shape automatically
                        if timesteps > 0:
                            expected_shape = (1, timesteps, len(ALL_FEATURES))
                        else:
                            expected_shape = (1, len(ALL_FEATURES))
                        
                        if model_input.shape != expected_shape:
                            print(f"WARNING: Input shape {model_input.shape} doesn't match expected {expected_shape}", file=sys.stderr)
                            print(f"Attempting prediction anyway - model will validate input shape...", file=sys.stderr)
                        
                        model = models['dual_mlp']
                        
                        # Time the model inference
                        t_inference_start = time.perf_counter()
                        
                        # Make prediction
                        if hasattr(model, 'get_input_details'):  # TFLite interpreter
                            input_details = model.get_input_details()
                            output_details = model.get_output_details()
                            
                            model.set_tensor(input_details[0]['index'], model_input.astype(np.float32))
                            model.invoke()
                            pred_nn_output = model.get_tensor(output_details[0]['index'])
                        else:  # Keras model
                            pred_nn_output = model(model_input, training=False).numpy()
                        
                        t_inference_end = time.perf_counter()
                        
                        # VALIDATION 5: Check output shape
                        if pred_nn_output.shape[1] != 2:
                            print(f"ERROR: Expected dual output with 2 values, got shape {pred_nn_output.shape}", file=sys.stderr)
                            set_flag(shm_map, CPP_WROTE_INPUT_OFFSET, False)
                            continue
                        
                        # Extract raw predictions (normalized space)
                        tilt_raw_output = pred_nn_output[0][0]  # First output is tilt
                        pan_raw_output = pred_nn_output[0][1]   # Second output is pan

                        previous_predictions = [tilt_raw_output, pan_raw_output]
                        #print(f"Raw predictions (normalized): Tilt={tilt_raw_output:.4f}, Pan={pan_raw_output:.4f}", file=sys.stderr)
                        # VALIDATION 6: Check if outputs are in reasonable range for normalized values
                        if abs(tilt_raw_output) > 10 or abs(pan_raw_output) > 10:
                            print(f"WARNING: Raw outputs seem unusually large (normalized): Tilt={tilt_raw_output:.3f}, Pan={pan_raw_output:.3f}", file=sys.stderr)
                        
                        # Denormalize using TiltCurrent and PanCurrent (the target variables)
                        # IMPORTANT: The model predicts the NEXT tick's current based on PREVIOUS timesteps
                        tilt_pred_scalar = float(tilt_raw_output * means_stds['dual_mlp']['TiltCurrent']['std'] + means_stds['dual_mlp']['TiltCurrent']['mean'])
                        pan_pred_scalar = float(pan_raw_output * means_stds['dual_mlp']['PanCurrent']['std'] + means_stds['dual_mlp']['PanCurrent']['mean'])
                        
                        predictions = [tilt_pred_scalar, pan_pred_scalar]
                        #print(f"Denormalized predictions: Tilt={tilt_pred_scalar:.4f}, Pan={pan_pred_scalar:.4f}", file=sys.stderr)
                        prediction_count += 1
                        # Get current actual values (for comparison and storage)
                        tilt_current_now = input_data_dict['servo_data'][0]['current']
                        pan_current_now = input_data_dict['servo_data'][1]['current']
                        
                        # VALIDATION 7: Detailed debugging for first 10 predictions and every 100th after
                        if hasattr(main, 'prediction_count'):
                            main.prediction_count += 1
                        else:
                            main.prediction_count = 1
                        
                        #should_debug = (main.prediction_count <= 10) or (main.prediction_count % 100 == 0)
                        
                        should_debug = False
                        if should_debug:
                            print(f"\n{'=' * 80}", file=sys.stderr)
                            print(f"PREDICTION #{main.prediction_count}", file=sys.stderr)
                            print(f"{'=' * 80}", file=sys.stderr)
                            
                            # Show raw input features (before normalization)
                            print("Raw input features (current tick t):", file=sys.stderr)
                            for feature, value in feature_debug:
                                print(f"  {feature:20s} = {value:8.4f}", file=sys.stderr)
                            
                            # Show a few normalized values
                            print("\nNormalized input (sample):", file=sys.stderr)
                            for i in range(min(4, len(ALL_FEATURES))):
                                raw_value = feature_debug[i][1] if i < len(feature_debug) else 0.0
                                # For debugging, show the back-calculated raw value from normalized
                                back_calc_raw = x_normalized[i] * (stds[i] + 1e-8) + means[i]
                                print(f"  {ALL_FEATURES[i]:20s} = {x_normalized[i]:8.4f} (normalized, back_calc_raw={back_calc_raw:8.4f}, mean={means[i]:8.4f}, std={stds[i]:8.4f})", file=sys.stderr)
                            
                            # Show model predictions
                            print(f"\nModel predictions (for NEXT tick t+1):", file=sys.stderr)
                            print(f"  Raw outputs (normalized): Tilt={tilt_raw_output:8.4f}, Pan={pan_raw_output:8.4f}", file=sys.stderr)
                            print(f"  Denormalization stats:", file=sys.stderr)
                            print(f"    TiltCurrent: mean={means_stds['dual_mlp']['TiltCurrent']['mean']:.4f}, std={means_stds['dual_mlp']['TiltCurrent']['std']:.4f}", file=sys.stderr)
                            print(f"    PanCurrent:  mean={means_stds['dual_mlp']['PanCurrent']['mean']:.4f}, std={means_stds['dual_mlp']['PanCurrent']['std']:.4f}", file=sys.stderr)
                            print(f"  Predicted (denormalized): Tilt={tilt_pred_scalar:8.4f}, Pan={pan_pred_scalar:8.4f}", file=sys.stderr)
                            
                            # Show current actual values
                            print(f"\nActual current values (current tick t):", file=sys.stderr)
                            print(f"  Tilt: {tilt_current_now:8.4f}", file=sys.stderr)
                            print(f"  Pan:  {pan_current_now:8.4f}", file=sys.stderr)
                            
                            # NOTE: We can't show the error here because we're predicting t+1 and only have t
                            print(f"\nNOTE: Prediction is for NEXT tick (t+1), actual shown is current tick (t)", file=sys.stderr)
                            print(f"      True prediction error will be visible in next tick's comparison", file=sys.stderr)
                            
                            # Show inference time
                            inference_time_ms = (t_inference_end - t_inference_start) * 1000
                            print(f"\nInference time: {inference_time_ms:.2f} ms", file=sys.stderr)
                            print(f"{'=' * 80}\n", file=sys.stderr)
                        
                        # Store predictions for comparison in next tick
                        if not hasattr(main, 'prev_predictions'):
                            main.prev_predictions = {'tilt': None, 'pan': None}
                            main.prev_actuals = {'tilt': None, 'pan': None}
                        
                        # Compare with previous prediction if available
                        if main.prev_predictions['tilt'] is not None and should_debug:
                            tilt_error = abs(main.prev_predictions['tilt'] - tilt_current_now)
                            pan_error = abs(main.prev_predictions['pan'] - pan_current_now)
                            
                            print(f"PREVIOUS PREDICTION ERROR CHECK:", file=sys.stderr)
                            print(f"  Tilt: predicted={main.prev_predictions['tilt']:8.4f}, actual={tilt_current_now:8.4f}, error={tilt_error:8.4f}", file=sys.stderr)
                            print(f"  Pan:  predicted={main.prev_predictions['pan']:8.4f}, actual={pan_current_now:8.4f}, error={pan_error:8.4f}", file=sys.stderr)
                            
                            if pan_error > 2 * tilt_error and pan_error > 0.1:
                                print(f"  WARNING: Pan error is significantly larger than Tilt error!", file=sys.stderr)
                        
                        # Store current predictions and actuals for next tick
                        main.prev_predictions['tilt'] = tilt_pred_scalar
                        main.prev_predictions['pan'] = pan_pred_scalar
                        main.prev_actuals['tilt'] = input_data_dict['servo_data'][0]['current']
                        main.prev_actuals['pan'] = input_data_dict['servo_data'][1]['current']
                        
                        # Update predicted current values for use in next prediction (if model uses current features)
                        # Use the RAW (normalized) predictions since they're in the same space as the training data
                        if has_current_features:
                            predicted_current_values['tilt'] = tilt_raw_output
                            predicted_current_values['pan'] = pan_raw_output
                            
                             
                    else:
                        # Original behavior for separate models per servo
                        raw_predictions = []  # Track raw (normalized) predictions for storing predicted currents
                        for i, servo_name_key in enumerate(servos): # tilt, pan
                            if servo_name_key == 'tilt':
                                feature_list = TILT_FEATURES
                            elif servo_name_key == 'pan':
                                feature_list = PAN_FEATURES
                            else:
                                raise ValueError(f"Unknown servo: {servo_name_key}")
                            
                            # Check if this servo model uses current features
                            has_current_features = any('Current' in feature for feature in feature_list)
                            # Use servo-specific flag for whether to use predicted current
                            use_predicted_current = has_current_features and use_prediction_as_input.get(servo_name_key, False)
                            
                            if "Output1" in model_name_suffix:
                                # For Output1, we assume a single output per servo
                                scalar_object = tilt_feature_scalar_object if servo_name_key == 'tilt' else pan_feature_scalar_object
                            else:
                                scalar_object = None
                            normalised_input = normalise_input(input_data_dict, means_stds[servo_name_key], servo_name_key, model_name_suffix, feature_list, scalar_object, use_predicted_current, predicted_current_values)

                            # Skip prediction if not enough data collected yet for temporal models
                            if normalised_input is None:
                                print(f"Waiting for more data for servo '{servo_name_key}' (temporal model needs {10 if 'tcn' in model_name_suffix.lower() else 5} timesteps)", file=sys.stderr)
                                # For temporal models, skip this entire cycle if any servo doesn't have enough data
                                predictions = []  # Clear any predictions from previous servos
                                break  # Exit the servo loop

                            #print(f"Normalised input for servo '{servo_name_key}': {normalised_input}", file=sys.stderr)
                            model = models[servo_name_key]
                            
                            # Time the model inference
                            t_inference_start = time.perf_counter()
                            
                            # Check if it's a TFLite interpreter or Keras model
                            if hasattr(model, 'get_input_details'):  # TFLite interpreter
                                input_details = model.get_input_details()
                                output_details = model.get_output_details()
                                
                                model.set_tensor(input_details[0]['index'], normalised_input.astype(np.float32))
                                model.invoke()
                                pred_nn_output = model.get_tensor(output_details[0]['index'])
                            else:  # Keras model
                                # Use __call__ instead of predict for faster inference (no overhead)
                                pred_nn_output = model(normalised_input, training=False).numpy()
                            
                            # Check and validate output shape, especially for TCN models
                            if 'tcn' in model_name_suffix.lower():
                                # TCN models should output shape (1, 1) for single prediction
                                if pred_nn_output.shape != (1, 1):
                                    print(f"Warning: TCN model for servo '{servo_name_key}' produced unexpected output shape {pred_nn_output.shape}, expected (1, 1)", file=sys.stderr)
                                    # Flatten if needed
                                    if pred_nn_output.size == 1:
                                        pred_nn_output = pred_nn_output.reshape(1, 1)
                                    else:
                                        # Take first element if multiple outputs
                                        pred_nn_output = np.array([[pred_nn_output.flatten()[0]]], dtype=np.float32)
                                        print(f"Reshaped to {pred_nn_output.shape} and using first output value", file=sys.stderr)
                            
                            # Ensure output is always 2D array with shape (1, 1) for consistency
                            if pred_nn_output.ndim == 1:
                                pred_nn_output = pred_nn_output.reshape(1, -1)
                            elif pred_nn_output.ndim == 0:
                                pred_nn_output = pred_nn_output.reshape(1, 1)
                            
                            # Log output shape occasionally for debugging
                            if hasattr(main, 'prediction_count') and main.prediction_count <= 3:
                                print(f"Servo '{servo_name_key}' model output shape: {pred_nn_output.shape}, value: {pred_nn_output[0][0]:.6f}", file=sys.stderr)
                            
                            t_inference_end = time.perf_counter()
                            inference_time_ms = (t_inference_end - t_inference_start) * 1000
                            
                            # Log timing occasionally (every 100th prediction)
                            if hasattr(main, 'prediction_count'):
                                main.prediction_count += 1
                            else:
                                main.prediction_count = 1
                            
                            # Store raw prediction for potential use as input in next iteration
                            raw_predictions.append(float(pred_nn_output[0][0]))
                            
                            target_var = f"{servo_name_key.capitalize()}Current"
                            
                            if "Output1" in model_name_suffix:
                                # For Output1, we assume a single output per servo
                                scalar_object = pan_y_scalar_object if servo_name_key == 'pan' else tilt_y_scalar_object
                                if scalar_object is None:
                                    raise ValueError(f"Scalar object for servo '{servo_name_key}' in model '{model_name_suffix}' could not be loaded. Check the scaler file path.")
                                pred_scalar = float(scalar_object.inverse_transform(pred_nn_output)[0][0])  # Assuming pred_nn_output is a 2D array with shape (1, 1)
                            else:    
                                pred_scalar = float(pred_nn_output[0][0] * means_stds[servo_name_key][target_var]['std'] + means_stds[servo_name_key][target_var]['mean'])
                            predictions.append(pred_scalar)
                        
                        # After all servo predictions are made, update predicted current values for next iteration
                        if len(predictions) == NUM_SERVOS and len(raw_predictions) == NUM_SERVOS:
                            # Update predicted current values if any servo model uses current features  
                            any_model_uses_current = any(any('Current' in feature for feature in (TILT_FEATURES if servo == 'tilt' else PAN_FEATURES)) for servo in servos)
                            if any_model_uses_current:
                                # Use raw (normalized) predictions for next iteration inputs
                                predicted_current_values['tilt'] = raw_predictions[0]  # Tilt raw prediction is first
                                predicted_current_values['pan'] = raw_predictions[1]   # Pan raw prediction is second
                                
                    
                    # Only write output if we have predictions for all servos
                    if len(predictions) == NUM_SERVOS:
                        t_start_write = time.perf_counter()
                        
                        # Write predictions back to shared memory
                        # First, the count of actual outputs
                        actual_output_count = len(predictions)
                        if actual_output_count > max_outputs_cpp:
                            print(f"Warning: Python produced {actual_output_count} outputs, but C++ expects max {max_outputs_cpp}. Truncating.", file=sys.stderr)
                            predictions = predictions[:max_outputs_cpp]
                            actual_output_count = max_outputs_cpp

                        # Pack count and then data
                        output_payload = struct.pack(f'<f{actual_output_count}f', float(actual_output_count), *predictions)
                        
                        shm_map.seek(output_data_offset)
                        shm_map.write(output_payload)
                        shm_map.flush()
                        
                        t_end_write = time.perf_counter()

                        # Signal C++ that output is ready
                        set_flag(shm_map, CPP_WROTE_INPUT_OFFSET, False) # C++ can write again
                        set_flag(shm_map, PYTHON_WROTE_OUTPUT_OFFSET, True) # Python has written

                        t_total = time.perf_counter() - t_start_read
                    else:
                        # Not enough data for all servos yet, just reset the input flag
                        set_flag(shm_map, CPP_WROTE_INPUT_OFFSET, False) # C++ can write again
                        # Don't set PYTHON_WROTE_OUTPUT_OFFSET to True since we have no output
               
                except Exception as e:
                    print(f"Error during prediction generation: {e}", file=sys.stderr)
                    # Reset flags defensively if an error occurs
                    try:
                        set_flag(shm_map, CPP_WROTE_INPUT_OFFSET, False) 
                        set_flag(shm_map, PYTHON_WROTE_OUTPUT_OFFSET, False) # Indicate no valid output
                    except Exception as reset_err:
                        print(f"Error resetting flags after prediction error: {reset_err}", file=sys.stderr)
            else:
                sleep(0.0001) # Small sleep to prevent CPU spinning
                
    except Exception as e:
        print(f"Error in main: {e}", file=sys.stderr)
        # Attempt to clean up even on main error
        if shm_map is not None: shm_map.close()
        if shm_handle is not None: shm_handle.close_fd()
        sys.exit(1)
    finally:
        print(f"Clean exit from ANN prediction service for SHM '{shm_name}'.", file=sys.stderr)
        if shm_map is not None:
            shm_map.close()
        if shm_handle is not None:
            shm_handle.close_fd()

if __name__ == "__main__":
    main()