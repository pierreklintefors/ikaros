import numpy as np
import mmap
import os
import json
from time import sleep
import sys
import signal
import struct  # Add struct for size calculations
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

# Update shared memory configuration for new servo-grouped data order
SHM_NAME = "/ikaros_ann_shm"
# New order: 
# - gyro(3)
# - accel(3)
# - angles(3)
# - For each servo (2 servos: tilt and pan):
#   - position(1)
#   - distance to goal(1)
#   - goal position(1)
#   - starting position(1)
# - predictions(4)
NUM_SERVOS = 2
# Important: Make sure FLOAT_COUNT matches the C++ definition
FLOAT_COUNT = 21  # 17 inputs + 4 predictions - must match C++ definition
# Calculate memory size using Python's struct module
FLOAT_SIZE = 4  # Float32 is 4 bytes
BOOL_SIZE = 1   # Boolean is 1 byte
MEM_SIZE = (FLOAT_COUNT * FLOAT_SIZE) + (BOOL_SIZE * 2)  # Match C++ calculation

class SharedMemory:
    def __init__(self):
        try:
            print(f"Attempting to open shared memory at {SHM_NAME}", file=sys.stderr)
            # Use posix_ipc to open shared memory
            self.memory = posix_ipc.SharedMemory(SHM_NAME)
            self.size = MEM_SIZE
            
            # Map the shared memory
            self.shared_data = mmap.mmap(self.memory.fd, self.size)
            print(f"Successfully mapped shared memory of size {self.size} bytes", 
                  file=sys.stderr)
                
        except Exception as e:
            print(f"Error initializing shared memory: {e}", file=sys.stderr)
            print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
            raise
            
    def __del__(self):
        try:
            if hasattr(self, 'shared_data') and self.shared_data is not None:
                self.shared_data.close()
                self.shared_data = None
                
            if hasattr(self, 'memory') and self.memory is not None:
                self.memory.close_fd()
                self.memory = None
        except Exception as e:
            print(f"Error cleaning up shared memory: {e}", file=sys.stderr)
    
    def read_data(self):
        self.shared_data.seek(0)
        data = np.frombuffer(self.shared_data.read(FLOAT_COUNT * 4), dtype=np.float32)
        
        
        # Extract data according to new servo-grouped order
        gyro = data[0:3]           # First 3 values are gyro
        accel = data[3:6]          # Next 3 values are accel
        angles = data[6:9]         # Next 3 values are euler angles
        
        # For each servo, extract its data
        num_servos = NUM_SERVOS  # tilt & pan
        servo_data_size = 4  # position, distance, goal, start
        
        positions = []
        distances = []
        goal_positions = []
        starting_positions = []
        
        for i in range(num_servos):
            # Calculate base index for this servo's data
            base_idx = 9 + (i * servo_data_size)
            
            
            positions.append(data[base_idx])
            distances.append(data[base_idx + 1])
            goal_positions.append(data[base_idx + 2])
            starting_positions.append(data[base_idx + 3])
       
        # Last 2 values are predictions
        predictions = data[-2:]
        # Restructure data to have all position data of one servo first, then the next servo
        servo_data = []
        for i in range(num_servos):
            servo_data.append({
                'position': positions[i],
                'distance': distances[i],
                'goal_position': goal_positions[i],
                'starting_position': starting_positions[i]
            })
        
        return {
            'gyro': gyro,
            'accel': accel,
            'angles': angles,
            'servo_data': servo_data,
            'predictions': predictions,
            'num_servos': num_servos
        }
    
    def write_prediction(self, predictions):
        # First, check and ensure we have the right number of predictions
        if len(predictions) > 4:  # We expect 4 values (2 per servo)
            print(f"Warning: Truncating predictions to 4 values (received {len(predictions)})", file=sys.stderr)
            predictions = predictions[:4]
        elif len(predictions) < 4:
            print(f"Warning: Padding predictions to 4 values (received {len(predictions)})", file=sys.stderr)
            predictions = predictions + [0.0] * (4 - len(predictions))
        
        print(f"Final predictions to write: {predictions}", file=sys.stderr)
        
        # Calculate the offset where predictions should be written
        prediction_offset = (FLOAT_COUNT - 4) * 4  # Skip past all input data
        
        # Write each prediction value individually to avoid array bounds issues
        try:
            for i, pred in enumerate(predictions):
                self.shared_data.seek(prediction_offset + (i * 4))
                self.shared_data.write(np.array([pred], dtype=np.float32).tobytes())
            print(f"Successfully wrote all predictions to offset {prediction_offset}", file=sys.stderr)
            
            # After writing predictions, reset the new_data flag to signal C++ that data is ready
            self.clear_new_data_flag()
            
        except Exception as e:
            print(f"Error writing predictions: {e}", file=sys.stderr)
            print(f"Offset: {prediction_offset}, Predictions: {predictions}", file=sys.stderr)
            raise
    
    def get_new_data_flag(self):
        # New data flag is stored at offset FLOAT_COUNT * 4
        self.shared_data.seek(FLOAT_COUNT * 4)
        flag_byte = self.shared_data.read(1)
        return bool(flag_byte[0])
    
    def clear_new_data_flag(self):
        # Explicitly reset the new_data flag to signal predictions are ready
        self.shared_data.seek(FLOAT_COUNT * 4)
        self.shared_data.write(bytearray([0]))
        print("Reset new_data flag to signal predictions are ready", file=sys.stderr)
    
    def get_shutdown_flag(self):
        # Shutdown flag is stored at offset (FLOAT_COUNT * 4) + 1
        self.shared_data.seek((FLOAT_COUNT * 4) + 1)
        flag_byte = self.shared_data.read(1)
        return bool(flag_byte[0])


def normalise_input(input_data, means_stds, servo_name, model_name):
    """
    Normalize input data for a specific servo model
    
    Args:
        input_data: Dictionary containing gyro, accel, positions, distances
        means_stds: Dictionary with normalization parameters
        servo_name: Name of the servo (e.g., 'tilt', 'pan')
    
    Returns:
        Normalized input as numpy array
    """
    # Define the expected feature order
    expected_feature_order = [
        'GyroX', 'GyroY', 'GyroZ', 'AccelX', 'AccelY', 'AccelZ', 'AngleX',
        'AngleY', 'AngleZ', 'TiltPosition', 'TiltDistToGoal',
        'TiltGoalPosition', 'TiltStartPosition', 'PanPosition', 'PanDistToGoal',
        'PanGoalPosition', 'PanStartPosition'
    ]
    
    # Get the list of available servos
    available_servos = [s for i, s in enumerate(['tilt', 'pan', 'roll', 'yaw', 'pitch']) 
                        if i < input_data['num_servos']]
    
    # Map servo names to their indices
    servo_indices = {name.lower(): i for i, name in enumerate(available_servos)}
    
    # Create input array in the expected order
    input_array = []
    extraction_log = []
    
    for feature in expected_feature_order:
        if feature not in means_stds:
            # Skip features that aren't in the means_stds dictionary
            continue
            
        feature_value = None
        
        if feature.startswith('Gyro'):
            axis = feature[-1]
            idx = {'X': 0, 'Y': 1, 'Z': 2}.get(axis, 0)
            if idx < len(input_data['gyro']):
                feature_value = input_data['gyro'][idx]
                extraction_log.append(f"{feature} = {feature_value} (gyro[{idx}])")
        
        elif feature.startswith('Accel'):
            axis = feature[-1]
            idx = {'X': 0, 'Y': 1, 'Z': 2}.get(axis, 0)
            if idx < len(input_data['accel']):
                feature_value = input_data['accel'][idx]
                extraction_log.append(f"{feature} = {feature_value} (accel[{idx}])")
        
        elif feature.startswith('Angle'):
            axis = feature[-1]
            idx = {'X': 0, 'Y': 1, 'Z': 2}.get(axis, 0)
            if idx < len(input_data['angles']):
                feature_value = input_data['angles'][idx]
                extraction_log.append(f"{feature} = {feature_value} (angles[{idx}])")
        
        else:
            # Extract servo name from feature (looking for standard prefixes)
            feature_servo_name = None
            for prefix in [s.capitalize() for s in available_servos]:
                if feature.startswith(prefix):
                    feature_servo_name = prefix.lower()
                    break
            
            if feature_servo_name is not None:
                servo_idx = servo_indices.get(feature_servo_name, -1)
                
                if servo_idx >= 0 and servo_idx < len(input_data['servo_data']):
                    # Get the specific value based on what kind of property it is
                    if feature.endswith('Position') and 'Goal' not in feature and 'Start' not in feature:
                        feature_value = input_data['servo_data'][servo_idx]['position']
                        extraction_log.append(f"{feature} = {feature_value} (servo_data[{servo_idx}]['position'])")
                    elif 'DistToGoal' in feature:
                        feature_value = input_data['servo_data'][servo_idx]['distance']
                        extraction_log.append(f"{feature} = {feature_value} (servo_data[{servo_idx}]['distance'])")
                    elif 'GoalPosition' in feature:
                        feature_value = input_data['servo_data'][servo_idx]['goal_position']
                        extraction_log.append(f"{feature} = {feature_value} (servo_data[{servo_idx}]['goal_position'])")
                    elif 'StartPosition' in feature:
                        feature_value = input_data['servo_data'][servo_idx]['starting_position']
                        extraction_log.append(f"{feature} = {feature_value} (servo_data[{servo_idx}]['starting_position'])")
        
        # If we couldn't extract a value, use 0.0 as fallback
        if feature_value is None:
            feature_value = 0.0
            extraction_log.append(f"{feature} = 0.0 (fallback)")
        
        input_array.append(feature_value)
    
    # Print detailed extraction log for debugging
    # print(f"Feature extraction log for {servo_name}:", file=sys.stderr)
    # for log_entry in extraction_log:
    #     print(f"  {log_entry}", file=sys.stderr)
    
    # Convert input_array to numpy array before normalization
    x = np.array(input_array)
    
    # Get means and stds for normalization - make sure to keep the same order as input_array
    means = np.array([means_stds[feature]['mean'] for feature in expected_feature_order if feature in means_stds])
    stds = np.array([means_stds[feature]['std'] for feature in expected_feature_order if feature in means_stds])
    
    # Ensure the input shape matches the expected shape
    if len(x) != len(means):
        print(f"Shape mismatch! Input shape: {x.shape}, Means shape: {means.shape}", file=sys.stderr)
        print(f"Input array: {x}", file=sys.stderr)
        print(f"Means: {means}", file=sys.stderr)
        print(f"Stds: {stds}", file=sys.stderr)
        
        # Pad or truncate input array to match expected size
        if len(x) < len(means):
            # Pad with zeros
            x = np.pad(x, (0, len(means) - len(x)), 'constant')
            print(f"Padded input to match expected size: {x.shape}", file=sys.stderr)
        else:
            # Truncate
            x = x[:len(means)]
            print(f"Truncated input to match expected size: {x.shape}", file=sys.stderr)
    
    #if modelname contains 'raw' then don't normalise
    if 'raw' in model_name:
        return x

    # Normalize
    return (x.reshape(1, -1) - means) / stds


def create_model_with_weights(weights_path, model_name, num_inputs):
    # Try to load model from config file
    config_path = weights_path.replace('.weights.h5', '_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model = Sequential()
            model.name = model_name
            
            # First layer needs input shape
            model.add(Input(shape=(num_inputs,)))  
            
            for layer in config['layers']:
                if layer['class_name'] == 'Dense':
                    layer_config = layer['config']
                    regularizer = None
                    if layer_config.get('kernel_regularizer'):
                        reg_config = layer_config['kernel_regularizer']['config']
                        regularizer = l2(reg_config.get('l2', 0))
                    
                    model.add(Dense(
                        units=layer_config['units'],
                        activation=layer_config['activation'],
                        kernel_regularizer=regularizer
                    ))
                elif layer['class_name'] == 'BatchNormalization':
                    model.add(BatchNormalization())
                elif layer['class_name'] == 'Dropout':
                    model.add(Dropout(layer['config']['rate']))
            
            # Compile model
            model.compile(optimizer=Adam(learning_rate=0.001), 
                          loss='mean_squared_error', 
                          metrics=['mae'])
            
            # Load pretrained weights
            model.load_weights(weights_path)
            
            print(f"Successfully loaded model from config: {config_path}", file=sys.stderr)
            return model
            
        except Exception as e:
            print(f"Error loading model from config: {e}", file=sys.stderr)
            print("Falling back to default model architecture", file=sys.stderr)
    
    # Fallback to default implementation
    print(f"Using default model architecture for {model_name}", file=sys.stderr)
    model = Sequential()
    model.name = model_name
    model.add(Input(shape=(num_inputs,)))  
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    model.load_weights(weights_path)
    return model


def signal_handler(sig, frame):
    print("Received termination signal. Cleaning up...", file=sys.stderr)
    # Global cleanup will happen via __del__ methods
    sys.exit(0)

def main():
    try:
        # Register signal handlers for proper cleanup
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        directory = os.path.dirname(os.path.abspath(__file__))
        
        servos = ['tilt', 'pan']  # This could be expanded for more servos
        #model_name = 'L1_128_L2_32_filtered_data_standardised_LeakyReLU_HuberLoss'
        model_name = 'L1_128_L2_32_filtered_data_standardised_Relu_MSE_2Output'
    # Load models and means/stds
        models = {}
        means_stds = {}
        for servo in servos:
            # Load model weights
            weights_path = directory + f'/weights/{servo}_{model_name}.weights.h5'
            models[servo] = create_model_with_weights(weights_path, f'{servo}_model', 17)
            
            # Load normalization parameters
            with open(directory + f'/weights/{servo}_filtered_data_mean_std.json', 'r') as f:
                means_stds[servo] = json.load(f)
        
        shm = SharedMemory()
        print("Successfully connected to shared memory", file=sys.stderr)
        
        while True:
            if shm.get_shutdown_flag():
                print("Shutdown flag received. Exiting ANN prediction loop.", file=sys.stderr)
                del shm  # Explicitly delete to ensure cleanup
                break

            if shm.get_new_data_flag():
                print("New data flag detected. Processing...", file=sys.stderr)
                
                # Read all input data
                input_data = shm.read_data()
       
                # Make predictions for each servo
                predictions = []
                try:
                    for i, servo in enumerate(servos):
                        # Normalize inputs for this specific servo
                        normalized_input = normalise_input(input_data, means_stds[servo], servo, model_name)
                        # Make prediction
                        pred = models[servo].predict(normalized_input, verbose=0)
                        
                        # Check if model has two outputs
                        if "2Output" in model_name:
                            # First output is the current
                            print(f"Predicted {servo} current: {pred[0][0]}", file=sys.stderr)
                            # Denormalize prediction for current
                            target_var = f"{servo.capitalize()}Current"
                            pred_scalar = abs(float(pred[0][0]) * means_stds[servo][target_var]['std'] + means_stds[servo][target_var]['mean'])
                            print(f"Denormalized {servo} current: {pred_scalar}", file=sys.stderr)
                            
                            # Second output is the effective current (not used in predictions list)
                            target_var_eff = f"{servo.capitalize()}EffectiveCurrent"
                            print(f"Predicted {servo} effective current: {pred[0][1]}", file=sys.stderr)
                            pred_eff_scalar = abs(float(pred[0][1]) * means_stds[servo][target_var_eff]['std'] + means_stds[servo][target_var_eff]['mean'])
                            print(f"Denormalized {servo} effective current: {pred_eff_scalar}", file=sys.stderr)
                            predictions.append(float(pred_scalar))    # Regular current
                            predictions.append(float(pred_eff_scalar)) # Start current
                        else:
                            # Single output model
                            print(f"Predicted {servo} current: {pred[0][0]}", file=sys.stderr)
                            # Denormalize prediction
                            target_var = f"{servo.capitalize()}Current"
                            pred_scalar = abs(float(pred[0][0]) * means_stds[servo][target_var]['std'] + means_stds[servo][target_var]['mean'])
                            print(f"Denormalized {servo} current: {pred_scalar}", file=sys.stderr)
                            
                            predictions.append(float(pred_scalar))
                    
                    # Debug before writing
                    print(f"Final prediction list before writing: {predictions}", file=sys.stderr)
                    print(f"Prediction types: {[type(p) for p in predictions]}", file=sys.stderr)
                    
                    # Write predictions - this will also clear the new_data flag
                    shm.write_prediction(predictions)
                    print(f"Predictions written to shared memory: {predictions}", file=sys.stderr)
                except Exception as e:
                    print(f"Error during prediction generation: {e}", file=sys.stderr)
                    # Even if there's an error, we should reset the new_data flag
                    try:
                        # Reset the new_data flag
                        shm.clear_new_data_flag()
                    except Exception as reset_err:
                        print(f"Error resetting new_data flag: {reset_err}", file=sys.stderr)
            else:
                # Small sleep to prevent CPU spinning when waiting for new data
                sleep(0.001)
                
    except Exception as e:
        print(f"Error in main: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        print("Clean exit from ANN prediction service", file=sys.stderr)

if __name__ == "__main__":
    main()