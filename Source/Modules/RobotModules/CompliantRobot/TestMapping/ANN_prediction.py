import numpy as np
import mmap
import os
import json
from time import sleep
import sys
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

# Update the shared memory path for macOS - remove /private/tmp/
SHM_NAME = "/ikaros_ann_shm"  # Changed this back
FLOAT_COUNT = 12  # 10 inputs + 2 predictions (tilt and pan)
MEM_SIZE = (FLOAT_COUNT * 4) + 2  # 4 bytes per float + 2 byte for  2 bool

class SharedMemory:
    def __init__(self):
        try:
            print(f"Attempting to open shared memory at {SHM_NAME}", file=sys.stderr)
            # Use posix_ipc to open shared memory
            self.memory = posix_ipc.SharedMemory(SHM_NAME)
            self.size = MEM_SIZE  # Use the same size calculation as C++
            
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
            if hasattr(self, 'shared_data'):
                self.shared_data.close()
            if hasattr(self, 'memory'):
                self.memory.close_fd()
        except Exception as e:
            print(f"Error cleaning up shared memory: {e}", file=sys.stderr)
    
    def read_data(self):
        self.shared_data.seek(0)
        data = np.frombuffer(self.shared_data.read(FLOAT_COUNT * 4), dtype=np.float32)
        
        # Determine the number of servos based on the data size
        # We know we have 3 gyro values, 3 accel values, and 2 values per servo (position + distance)
        num_servos = (len(data) - 6 - 2) // 2  # Subtract gyro, accel, and predictions, divide by 2
        
        # Extract data dynamically
        gyro = data[0:3]  # First 3 values are gyro
        accel = data[3:6]  # Next 3 values are accel
        
        # Extract positions and distances for each servo
        positions = []
        distances = []
        for i in range(num_servos):
            positions.append(data[6 + i*2])      # Position
            distances.append(data[6 + i*2 + 1])  # Distance to goal
        
        # Last 2 values are predictions
        predictions = data[-2:]
        
        return {
            'gyro': gyro,
            'accel': accel,
            'positions': positions,
            'distances': distances,
            'predictions': predictions,
            'num_servos': num_servos
        }
    
    def write_prediction(self, predictions):
        # Write predictions starting at the correct offset
        self.shared_data.seek(10 * 4)  # Skip past input data
        self.shared_data.write(np.array(predictions, dtype=np.float32).tobytes())
        # Don't clear new_data flag here anymore - let C++ do it after reading
    
    def get_new_data_flag(self):
        # New data flag is stored at offset FLOAT_COUNT * 4
        self.shared_data.seek(FLOAT_COUNT * 4)
        flag_byte = self.shared_data.read(1)
        return bool(flag_byte[0])
    
    def get_shutdown_flag(self):
        # Shutdown flag is stored at offset (FLOAT_COUNT * 4) + 1
        self.shared_data.seek((FLOAT_COUNT * 4) + 1)
        flag_byte = self.shared_data.read(1)
        return bool(flag_byte[0])


def normalise_input(input_data, means_stds, servo_name):
    """
    Normalize input data for a specific servo model
    
    Args:
        input_data: Dictionary containing gyro, accel, positions, distances
        means_stds: Dictionary with normalization parameters
        servo_name: Name of the servo (e.g., 'tilt', 'pan')
    
    Returns:
        Normalized input as numpy array
    """
    # Get the feature order from the means_stds dictionary
    feature_names = list(means_stds.keys())
    # Remove the target variable (e.g., 'TiltCurrent' or 'PanCurrent')
    feature_names = [name for name in feature_names if not name.endswith('Current')]
    
    # Create input array in the same order as feature_names
    input_array = []
    for feature in feature_names:
        if feature.startswith('Gyro'):
            idx = {'GyroX': 0, 'GyroY': 1, 'GyroZ': 2}.get(feature, 0)
            input_array.append(input_data['gyro'][idx])
        elif feature.startswith('Accel'):
            idx = {'AccelX': 0, 'AccelY': 1, 'AccelZ': 2}.get(feature, 0)
            input_array.append(input_data['accel'][idx])
        elif feature.endswith('Position'):
            if feature.startswith('Tilt'):
                input_array.append(input_data['positions'][0])
            elif feature.startswith('Pan'):
                input_array.append(input_data['positions'][1])
        elif feature.endswith('DistToGoal'):
            if feature.startswith('Tilt'):
                input_array.append(input_data['distances'][0])
            elif feature.startswith('Pan'):
                input_array.append(input_data['distances'][1])
    
    # Convert to numpy array
    x = np.array(input_array)
    
    # Get means and stds for normalization
    means = np.array([means_stds[name]['mean'] for name in feature_names])
    stds = np.array([means_stds[name]['std'] for name in feature_names])
    
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
            model.add(Input(shape=(10,)))  
            
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
    model.add(Input(shape=(num_inputs,)))  # Changed from 11 to 10 to match actual input size
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


def main():
    try:
        directory = os.path.dirname(os.path.abspath(__file__))
        
        servos = ['tilt', 'pan']  # This could be expanded for more servos

        # Load models and means/stds
        models = {}
        means_stds = {}
        for servo in servos:
            # Load model weights
            weights_path = directory + f'/weights/{servo}_filtered_model.weights.h5'
            models[servo] = create_model_with_weights(weights_path, f'{servo}_model', 10)
            
            # Load normalization parameters
            with open(directory + f'/weights/{servo}_mean_std.json', 'r') as f:
                means_stds[servo] = json.load(f)
        
        shm = SharedMemory()
        print("Successfully connected to shared memory", file=sys.stderr)
        
        while True:
            if shm.get_shutdown_flag():
                print("Shutdown flag received. Exiting ANN prediction loop.", file=sys.stderr)
                break

            if shm.get_new_data_flag():
                # Read all input data
                input_data = shm.read_data()
                
                # Make predictions for each servo
                predictions = []
                for i, servo in enumerate(servos):
                    # Normalize inputs for this specific servo
                    normalized_input = normalise_input(input_data, means_stds[servo], servo)
                    
                    # Make prediction
                    pred = models[servo].predict(normalized_input, verbose=0)
                    
                    # Denormalize prediction
                    target_var = f"{servo.capitalize()}Current"
                    pred_scalar = float(pred[0][0]) * means_stds[servo][target_var]['std'] + means_stds[servo][target_var]['mean']
                    predictions.append(pred_scalar)
                    
                    print(f"Input data: {normalized_input}")
                    print(f"{servo}_pred", pred)
                    print(f"{servo}_pred_scalar", pred_scalar)
                
                # Write predictions
                shm.write_prediction(predictions)
                
    except Exception as e:
        print(f"Error in main: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()