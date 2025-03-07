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
        data = np.frombuffer(self.shared_data.read(12 * 4), dtype=np.float32)
        # First 2 are positions (tilt, pan)
        # Next 3 are gyro
        # Next 3 are accel
        # Next 2 are distances to goal
        # Last 2 are for predictions (tilt, pan)
        return {
            'positions': data[:2],
            'gyro': data[2:5],
            'accel': data[5:8],
            'distances': data[8:10],
            'predictions': data[10:]
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


def normalise_input(x, means_stds):
    # Create array of all means and standard deviations in correct order
    feature_names = ['TiltPosition', 'PanPosition', 'GyroX', 'GyroY', 'GyroZ', 
                     'AccelX', 'AccelY', 'AccelZ', 'TiltDistToGoal', 'PanDistToGoal']
    
    means = np.array([means_stds[name]['mean'] for name in feature_names])
    stds = np.array([means_stds[name]['std'] for name in feature_names])
    
    # Vectorized normalization
    return (x.reshape(1, -1) - means) / stds

def create_model_with_weights(weights_path, model_name):
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
    model.add(Input(shape=(10,)))  # Changed from 11 to 10 to match actual input size
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
        
        servos = ['tilt', 'pan']

        # Load models and means/stds
        models = {}
        means_stds = {}
        for servo in servos:
            models[servo] = create_model_with_weights(directory + f'/weights/{servo}_filtered_model.weights.h5', f'{servo}_model')
            with open(directory + f'/weights/{servo}_mean_std.json', 'r') as f:
                means_stds[servo] = json.load(f)


        
        shm = SharedMemory()  # This will now use the correct SHM_NAME
        print("Successfully connected to shared memory", file=sys.stderr)
        
        while True:
            if shm.get_shutdown_flag():
                print("Shutdown flag received. Exiting ANN prediction loop.", file=sys.stderr)
                break

            if shm.get_new_data_flag():
                # Read all input data
                input_data = shm.read_data()
                
                # Create input array for models
                model_inputs = np.concatenate([
                    input_data['positions'],
                    input_data['gyro'],
                    input_data['accel'],
                    input_data['distances']
                ]).reshape(1, -1)
           
                # Normalize inputs
                tilt_normalized = normalise_input(model_inputs[0], means_stds['tilt'])
                pan_normalized = normalise_input(model_inputs[0], means_stds['pan'])
                
                # Make predictions
                tilt_pred = models['tilt'].predict(tilt_normalized, verbose=0)
                pan_pred = models['pan'].predict(pan_normalized, verbose=0)
                
                # Denormalize predictions - extract scalar values properly
                tilt_pred_scalar = float(tilt_pred[0][0]) * means_stds['tilt']['TiltCurrent']['std'] + means_stds['tilt']['TiltCurrent']['mean'] 
                pan_pred_scalar = float(pan_pred[0][0]) * means_stds['pan']['PanCurrent']['std'] + means_stds['pan']['PanCurrent']['mean'] 
                
                # Write predictions without clearing the flag
                predictions = [tilt_pred_scalar, pan_pred_scalar]
                shm.write_prediction(predictions)
                
         
            
    except Exception as e:
        print(f"Error in main: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()