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
try:
    import posix_ipc
except ImportError as e:
    print(f"Error importing posix_ipc: {e}", file=sys.stderr)
    print(f"Current environment: {os.environ.get('VIRTUAL_ENV')}", file=sys.stderr)
    sys.exit(1)
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, Input
from keras.regularizers import l2
from keras.optimizers import Adam

import tensorflow as tf
import os # Make sure os is imported if not already

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
EXPECTED_INPUTS_PER_SERVO = 5
EXPECTED_BASE_INPUTS = 9 # Gyro(3) + Accel(3) + Angles(3)
EXPECTED_TOTAL_INPUTS_FROM_CPP = EXPECTED_BASE_INPUTS + NUM_SERVOS * EXPECTED_INPUTS_PER_SERVO


# Define which features to use for each servo model
# TILT_FEATURES = ['GyroX', 'GyroY', 'GyroZ', 'TiltPosition', 'TiltDistToGoal', 'TiltDistToStart']
# PAN_FEATURES = ['PanPosition', 'PanDistToGoal', 'PanDistToStart', 'TiltPosition', 'TiltDistToGoal', 'TiltDistToStart']

# Global constants for file paths and model configuration
directory = os.path.dirname(os.path.abspath(__file__))
servos = ['tilt', 'pan'] 
#model_name_suffix = 'L1_128_L2_64_position_control_standardised_Relu_mean_squared_error_2Output_DropOut_0.2_l2_0.001_all_data'
#mean_stds_suffix = 'all_data_position_control_mean_std.json'
#model_name_suffix = 'position_control_Output1'
#mean_stds_suffix = 'mean_std_position_control_Output1.json'
model_name_suffix = 'L1_256_L2_32_position_control_standardised_Relu_mean_squared_error_1Output_DropOut_0.15_l2_0.003_all_data_small_set'
mean_stds_suffix = 'small_data_position_control_mean_std.json'
if "Output1" in model_name_suffix:
    with open(directory + f'/weights/scalers/scaler_X_pan_position_control_Output1.pkl', 'rb') as f:
        pan_feature_scalar_object = pickle.load(f)
    with open(directory + f'/weights/scalers/scaler_X_tilt_position_control_Output1.pkl', 'rb') as f:
        tilt_feature_scalar_object = pickle.load(f)
    with open(directory + f'/weights/scalers/scaler_y_pan_position_control_Output1.pkl', 'rb') as f:
        pan_y_scalar_object = pickle.load(f)
    with open(directory + f'/weights/scalers/scaler_y_tilt_position_control_Output1.pkl', 'rb') as f:
        tilt_y_scalar_object = pickle.load(f)
else:
    pan_feature_scalar_object = None
    tilt_feature_scalar_object = None
    pan_y_scalar_object = None
    tilt_y_scalar_object = None
        
TILT_FEATURES = []
PAN_FEATURES = []

for servo in servos:
    if servo == 'tilt':
        with open(directory + f'/weights/{servo}_{model_name_suffix}_config.json', 'r') as f:
            config = json.load(f)
            TILT_FEATURES = config['features']
        # Load the features from the config file
    elif servo == 'pan':
        with open(directory + f'/weights/{servo}_{model_name_suffix}_config.json', 'r') as f:
            config = json.load(f)
            PAN_FEATURES = config['features']
    else:
        raise ValueError(f"Unknown servo: {servo}")
print(f"Tilt features: {TILT_FEATURES}", file=sys.stderr)
print(f"Pan features: {PAN_FEATURES}", file=sys.stderr)

# Global variable for shared memory map, to be cleaned up in signal_handler
shm_map = None
shm_handle = None

def get_feature_value(input_data, feature, servo_name):
    """
    Extracts the value of a specific feature from the input data dictionary.
    Handles both servo-specific features and common IMU features.
    """
    # Map from feature name to input_data['servo_data'] keys
    servo_feature_map = {
        'Position': 'position',
        'DistToGoal': 'goal_distance',
        'DistToStart': 'start_distance',
        'GoalPosition': 'goal_position',
        'StartPosition': 'starting_position'
    }

    # Handle IMU features
    if feature in ['GyroX', 'GyroY', 'GyroZ']:
        return input_data['gyro'][['GyroX', 'GyroY', 'GyroZ'].index(feature)]
    if feature in ['AccelX', 'AccelY', 'AccelZ']:
        return input_data['accel'][['AccelX', 'AccelY', 'AccelZ'].index(feature)]
    if feature in ['AngleX', 'AngleY', 'AngleZ']:
        return input_data['angles'][['AngleX', 'AngleY', 'AngleZ'].index(feature)]

    # Handle servo-specific features (e.g., TiltPosition, PanDistToGoal)
    for prefix in ['Tilt', 'Pan']:
        if feature.startswith(prefix):
            # Determine which servo index to use
            servo_idx = 0 if prefix == 'Tilt' else 1
            
            # Extract the metric from the feature name
            # e.g., 'TiltPosition' -> 'Position'
            metric = feature[len(prefix):]
            
            # Check if the metric is in our map
            if metric in servo_feature_map:
                # Get the key for the servo_data dictionary
                data_key = servo_feature_map[metric]
                # Return the value from the correct servo's data
                return input_data['servo_data'][servo_idx][data_key]
    
    # If not found, raise error
    raise ValueError(f"Unknown feature: {feature}")

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

    features_per_servo = EXPECTED_INPUTS_PER_SERVO
    for i in range(num_servos_expected):
        pos = flat_input_array[ptr]; ptr += 1
        dist_goal = flat_input_array[ptr]; ptr += 1
        dist_start = flat_input_array[ptr]; ptr += 1
        goal_pos = flat_input_array[ptr]; ptr += 1
        start_pos = flat_input_array[ptr]; ptr += 1

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
            'starting_position': start_pos
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
        # 'predictions' key is not part of input, it was an old artifact
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

def normalise_input(input_data, means_stds, servo_name, model_name, feature_list, scalar_object=None):
    """
    Normalize input data for a specific servo model using a custom feature list.
    """
    input_array = []
    for feature in feature_list:
        if feature not in means_stds:
            # It's possible for a feature to be in the feature_list (e.g. common features like GyroX for Pan model)
            # but not have its own entry in means_stds[servo_name] if it's meant to use global or another servo's stats.
            # However, the current structure implies means_stds[servo_name] should contain all needed stats for that servo's model.
            raise ValueError(f"Feature '{feature}' not found in means_stds for servo '{servo_name}', model '{model_name}'. Check your mean_std.json files and feature lists.")

        feature_value = None
        processed = False # Flag to track if feature_value was set by specific logic

        # Servo-specific features (e.g., TiltPosition, PanDistToGoal)
        # These features combine the servo name with the metric.
        current_servo_prefix = servo_name.capitalize() # "Tilt" or "Pan"
        
        if feature.startswith(current_servo_prefix):
            servo_idx_in_data = 0 if servo_name == 'tilt' else 1 # 0 for tilt, 1 for pan in input_data['servo_data']
            
            if feature == f"{current_servo_prefix}Position":
                feature_value = input_data['servo_data'][servo_idx_in_data]['position']
                processed = True
            elif feature == f"{current_servo_prefix}DistToGoal":
                feature_value = input_data['servo_data'][servo_idx_in_data]['goal_distance']
                processed = True
            elif feature == f"{current_servo_prefix}DistToStart":
                feature_value = input_data['servo_data'][servo_idx_in_data]['start_distance']
                processed = True
            elif feature == f"{current_servo_prefix}GoalPosition":
                feature_value = input_data['servo_data'][servo_idx_in_data]['goal_position']
                processed = True
            elif feature == f"{current_servo_prefix}StartPosition":
                feature_value = input_data['servo_data'][servo_idx_in_data]['starting_position']
                processed = True

        # Handling for features that might be from the OTHER servo (e.g., Pan model using TiltPosition)
        # This is relevant for PAN_FEATURES which includes TiltPosition, TiltDistToGoal, TiltDistToStart
        elif servo_name == 'pan' and feature.startswith('Tilt'):
            other_servo_idx_in_data = 0 # Tilt is always index 0 in input_data['servo_data']
            if feature == 'TiltPosition':
                feature_value = input_data['servo_data'][other_servo_idx_in_data]['position']
                processed = True
            elif feature == 'TiltDistToGoal':
                feature_value = input_data['servo_data'][other_servo_idx_in_data]['goal_distance']
                processed = True
            elif feature == 'TiltDistToStart':
                feature_value = input_data['servo_data'][other_servo_idx_in_data]['start_distance']
                processed = True
        
        # IMU features (common to both models or general)
        if not processed:
            if feature == 'GyroX':
                feature_value = input_data['gyro'][0]
                processed = True
            elif feature == 'GyroY':
                feature_value = input_data['gyro'][1]
                processed = True
            elif feature == 'GyroZ':
                feature_value = input_data['gyro'][2]
                processed = True
            elif feature == 'AccelX':
                feature_value = input_data['accel'][0]
                processed = True
            elif feature == 'AccelY':
                feature_value = input_data['accel'][1]
                processed = True
            elif feature == 'AccelZ':
                feature_value = input_data['accel'][2]
                processed = True
            elif feature == 'AngleX':
                feature_value = input_data['angles'][0]
                processed = True
            elif feature == 'AngleY':
                feature_value = input_data['angles'][1]
                processed = True
            elif feature == 'AngleZ':
                feature_value = input_data['angles'][2]
                processed = True

        if not processed:
            # If feature_value is still None, it means the feature in feature_list
            # was not handled by any of the specific extraction logic above.
            # This indicates a mismatch between feature_list and extraction logic.
            # The original code had a fallback to mean/default, but it's safer to error out
            # or at least be very explicit if a feature is expected but not found.
            # For robustness, let's try the default from original code but with a clear warning.
            print(f"Warning: Feature '{feature}' for servo '{servo_name}' was not explicitly extracted from input_data. Attempting to use default value from means_stds.", file=sys.stderr)
            if means_stds[feature].get('default') is not None:
                    feature_value = means_stds[feature]['default']
            elif means_stds[feature].get('mean') is not None: # Fallback to mean if default is not present
                    feature_value = means_stds[feature]['mean']
            else:
                    raise ValueError(f"Feature '{feature}' for servo '{servo_name}' could not be extracted and no 'default' or 'mean' found in means_stds.")
            processed = True # Mark as processed via fallback

        if feature_value is None and processed is False: # Should ideally not happen if logic is complete
                raise ValueError(f"Critical: Feature '{feature}' for servo '{servo_name}' was not processed and resulted in a None value before appending to input_array.")

        input_array.append(feature_value)

    x = np.array(input_array)

    if scalar_object is not None:
        if "Output1" in model_name:
            print(f"Normalizing input with scalar object: {model_name}", file=sys.stderr)
            # Build a DataFrame with feature names for the scaler
            x_df = pd.DataFrame([dict(zip(feature_list, x.flatten()))])
            if not all(feature in scalar_object.feature_names_in_ for feature in feature_list):
                raise ValueError(f"Feature list {feature_list} does not match scalar object features {scalar_object.feature_names_in_} for servo '{servo_name}' in model '{model_name}'.")
            x = scalar_object.transform(x_df) 
            return x
    else:
        means = np.array([means_stds[feature]['mean'] for feature in feature_list])
        stds = np.array([means_stds[feature]['std'] for feature in feature_list])
        stds = np.maximum(stds, 1e-2) # Avoid division by zero or very small stds

    if 'raw' in model_name:
        return x.reshape(1, -1) # Return raw data, but ensure it's 2D

    return (x.reshape(1, -1) - means) / stds


def create_model_with_weights(weights_path, model_name, num_inputs):
    # Importing json here as it's only used in this function and main
    import json 
    tflite_model_path = weights_path.replace('.weights.h5', '.tflite')

    # Try to load TFLite model if it exists
    if os.path.exists(tflite_model_path):
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
                elif layer_config_item['class_name'] == 'BatchNormalization':
                    keras_model.add(BatchNormalization())
                elif layer_config_item['class_name'] == 'Dropout':
                    keras_model.add(Dropout(layer_config_item['config']['rate']))
            
            keras_model.compile(optimizer=Adam(learning_rate=0.001), 
                          loss='mean_squared_error', 
                          metrics=['mae'])
            
            keras_model.load_weights(weights_path)
            print(f"Successfully loaded Keras model from config: {config_path} for TFLite conversion", file=sys.stderr)
            
        except Exception as e:
            print(f"Error loading Keras model from config for TFLite conversion: {e}", file=sys.stderr)
            keras_model = None # Reset to ensure fallback is used if config loading fails partially

    if keras_model is None: # Fallback to default Keras model architecture if config failed or no weights_path
        print(f"Using default Keras model architecture for {model_name} (or weights_path was None) for TFLite conversion", file=sys.stderr)
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
            print(f"Warning: No Keras weights file found at {weights_path}. Model will be uninitialized if not for TFLite conversion from scratch.", file=sys.stderr)


    # Convert Keras model to TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        # You can enable optimizations like float16 quantization here if desired
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
        tflite_model_content = converter.convert()
        
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model_content)
        print(f"Successfully converted Keras model to TFLite and saved to {tflite_model_path}", file=sys.stderr)
        
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path) # Load from file to be consistent
        # Or use: interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error converting Keras model to TFLite: {e}", file=sys.stderr)
        raise # Re-raise the exception as we can't proceed without a model


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

        if num_inputs_cpp != EXPECTED_TOTAL_INPUTS_FROM_CPP:
            print(f"Error: Received {num_inputs_cpp} inputs from C++, but script expects {EXPECTED_TOTAL_INPUTS_FROM_CPP} for {NUM_SERVOS} servos.", file=sys.stderr)
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

        for servo in servos:
            feature_list_for_servo = TILT_FEATURES if servo == 'tilt' else PAN_FEATURES
            num_inputs_for_this_model = len(feature_list_for_servo)
            if num_features_for_model_init == 0: # Store for dummy input
                 num_features_for_model_init = num_inputs_for_this_model

            weights_path = directory + f'/weights/{servo}_{model_name_suffix}.weights.h5'
            models[servo] = create_model_with_weights(weights_path, f'{servo}_model', num_inputs_for_this_model)
            
            with open(directory + f'/weights/{servo}_{mean_stds_suffix}') as f:
                means_stds[servo] = json.load(f)
        
        if num_features_for_model_init > 0:
            dummy_input = np.zeros((1, num_features_for_model_init))
            for servo in servos:
                interpreter = models[servo]
                input_details = interpreter.get_input_details()
                interpreter.set_tensor(input_details[0]['index'], dummy_input.astype(np.float32))
                interpreter.invoke()
                _ = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            print("Models warmed up.", file=sys.stderr)
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
                t_start_predict_loop = time.perf_counter()
                try:
                    for i, servo_name_key in enumerate(servos): # tilt, pan
                        # ... (rest of prediction logic using input_data_dict)
                        if servo_name_key == 'tilt':
                            feature_list = TILT_FEATURES
                        elif servo_name_key == 'pan':
                            feature_list = PAN_FEATURES
                        else:
                            raise ValueError(f"Unknown servo: {servo_name_key}")
                        
                        if "Output1" in model_name_suffix:
                            # For Output1, we assume a single output per servo
                            scalar_object = tilt_feature_scalar_object if servo_name_key == 'tilt' else pan_feature_scalar_object
                        else:
                            scalar_object = None
                        normalized_input = normalise_input(input_data_dict, means_stds[servo_name_key], servo_name_key, model_name_suffix, feature_list, scalar_object)
  
                        interpreter = models[servo_name_key]
                        input_details = interpreter.get_input_details()
                        output_details = interpreter.get_output_details()
                        
                        interpreter.set_tensor(input_details[0]['index'], normalized_input.astype(np.float32))
                        interpreter.invoke()
                        pred_nn_output = interpreter.get_tensor(output_details[0]['index'])

                      
                        
                        target_var = f"{servo_name_key.capitalize()}Current"
                        
                        if "Output1" in model_name_suffix:
                            # For Output1, we assume a single output per servo
                            scalar_object = pan_y_scalar_object if servo_name_key == 'pan' else tilt_y_scalar_object
                            if scalar_object is None:
                                raise ValueError(f"Scalar object for servo '{servo_name_key}' in model '{model_name_suffix}' could not be loaded. Check the scaler file path.")
                            #pred_df = pd.DataFrame(pred_nn_output, columns=scalar_object.feature_names_in_)
                            #pred_scalar = float(scalar_object.inverse_transform(pred_df)[0][0])
                            pred_scalar = float(scalar_object.inverse_transform(pred_nn_output)[0][0])  # Assuming pred_nn_output is a 2D array with shape (1, 1)

                        else:    
                            pred_scalar = float(pred_nn_output[0][0] * means_stds[servo_name_key][target_var]['std'] + means_stds[servo_name_key][target_var]['mean'])
                        predictions.append(pred_scalar)
                    
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
                    # print(f"Total Python time: {t_total:.6f}s (Read={t_end_read-t_start_read:.6f}s, PredictLoop={t_start_write-t_start_predict_loop:.6f}s, Write={t_end_write-t_start_write:.6f}s)", file=sys.stderr)
                    # print("Predictions sent: ", predictions, file=sys.stderr)

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