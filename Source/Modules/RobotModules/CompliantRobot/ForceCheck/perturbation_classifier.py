#!/usr/bin/env python3
"""
Sequence-Based Perturbation Classification Script for ForceCheck
Communicates via shared memory with PythonScriptCaller.cc

This classifier uses a sequence-based Random Forest model that aggregates
features over a window of time points to classify perturbations.

Input format (4 floats per timestep):
  [0] tilt_current
  [1] pan_current  
  [2] tilt_prediction
  [3] pan_prediction

Output format (2 floats):
  [0] class_index (0=none, 1=obstacle, 2=push, 3=sustained)
  [1] confidence (0.0-1.0)
"""

import sys
import os
import pickle
import numpy as np
import posix_ipc
import mmap
import struct
import signal
from collections import deque

# Global flag for shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    global shutdown_flag
    shutdown_flag = True
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def create_sequence_features(sequence_data, feature_names):
    """
    Create aggregated features from a sequence of data points.
    
    Args:
        sequence_data: numpy array of shape (sequence_length, num_features)
        feature_names: list of feature column names
    
    Returns:
        Dictionary of aggregated features
    """
    features = {}
    
    for i, col_name in enumerate(feature_names):
        col_data = sequence_data[:, i]
        
        # Statistical aggregations
        features[f'{col_name}_mean'] = np.mean(col_data)
        features[f'{col_name}_std'] = np.std(col_data)
        features[f'{col_name}_min'] = np.min(col_data)
        features[f'{col_name}_max'] = np.max(col_data)
        features[f'{col_name}_range'] = np.max(col_data) - np.min(col_data)
        features[f'{col_name}_median'] = np.median(col_data)
        
        # Temporal features
        features[f'{col_name}_first'] = col_data[0]
        features[f'{col_name}_last'] = col_data[-1]
        features[f'{col_name}_diff_first_last'] = col_data[-1] - col_data[0]
        
        # Peak detection
        features[f'{col_name}_abs_max'] = np.abs(col_data).max()
        features[f'{col_name}_abs_mean'] = np.abs(col_data).mean()
    
    # NOTE: NOT including sequence_length - it would be constant during inference
    # and shouldn't influence classification
    
    return features

def main():
    if len(sys.argv) != 5:
        print("Usage: <shm_name> <num_inputs> <num_outputs> <flags_size>", file=sys.stderr)
        sys.exit(1)
    
    shm_name = sys.argv[1]
    num_inputs = int(sys.argv[2])
    num_outputs = int(sys.argv[3])
    flags_size = int(sys.argv[4])
    
    # Load sliding window model, scaler, and label encoder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'sliding_window_classifier_rf.pkl')
    scaler_path = os.path.join(script_dir, 'models', 'sliding_window_scaler.pkl')
    encoder_path = os.path.join(script_dir, 'models', 'sliding_window_label_encoder.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Model loaded: {model.__class__.__name__}", file=sys.stderr)
    print(f"Classes: {label_encoder.classes_}", file=sys.stderr)
    print(f"Number of trees: {len(model.estimators_)}", file=sys.stderr)
    
    # Feature names (raw features before aggregation)
    raw_feature_names = ['tilt_current', 'pan_current', 'tilt_error', 'pan_error']
    
    # Get expected feature names from the model (for proper ordering)
    # The scaler should have the feature names if trained on a DataFrame
    # Otherwise, we construct them based on the aggregation function
    
    # Build expected feature names in the same order as during training
    # NOTE: NOT including sequence_length - window size is fixed during inference
    expected_features = []
    for feat in raw_feature_names:
        expected_features.extend([
            f'{feat}_mean', f'{feat}_std', f'{feat}_min', f'{feat}_max',
            f'{feat}_range', f'{feat}_median', f'{feat}_first', f'{feat}_last',
            f'{feat}_diff_first_last', f'{feat}_abs_max', f'{feat}_abs_mean'
        ])
    
    print(f"Expected number of features: {len(expected_features)}", file=sys.stderr)
    
    # Open shared memory
    shm_name_with_slash = "/" + shm_name if not shm_name.startswith("/") else shm_name
    shm = posix_ipc.SharedMemory(shm_name_with_slash)
    
    data_array_size = (num_inputs + 1 + num_outputs) * 4  # floats
    total_size = flags_size + data_array_size
    
    mm = mmap.mmap(shm.fd, total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    
    print("Sequence-based perturbation classifier ready", file=sys.stderr)
    sys.stderr.flush()
    
    # Pre-calculate struct formats
    flags_format = '???'
    input_format = f'{num_inputs}f'
    output_format = 'fff'  # count + 2 outputs
    
    # Sequence buffer configuration
    # Model was trained on sliding windows of fixed size 50
    # This matches the inference scenario perfectly - no train/inference mismatch!
    WINDOW_SIZE = 20  # Must match training window size
    sequence_buffer = deque(maxlen=WINDOW_SIZE)
    
    # Cache for avoiding repeated predictions on similar inputs
    last_prediction = 0
    last_confidence = 1.0
    prediction_counter = 0
    PREDICTION_INTERVAL = 5  # Make prediction every N samples (balance speed vs responsiveness)
    
    while not shutdown_flag:
        # Read flags and input data in single operation for efficiency
        mm.seek(0)
        flags_and_data = mm.read(flags_size + num_inputs * 4)
        
        cpp_wrote_input, python_wrote_output, shutdown_signal = struct.unpack_from(flags_format, flags_and_data, 0)
        
        if shutdown_signal:
            break
        
        if cpp_wrote_input and not python_wrote_output:
            # Unpack input data from the already-read buffer
            inputs = struct.unpack_from(input_format, flags_and_data, flags_size)
            
            # inputs = [tilt_current, pan_current, tilt_prediction, pan_prediction]
            # Calculate errors
            tilt_current = inputs[0]
            pan_current = inputs[1]
            tilt_error = inputs[2] - inputs[0]
            pan_error = inputs[3] - inputs[1]
            
            # Add to sequence buffer
            sequence_buffer.append([tilt_current, pan_current, tilt_error, pan_error])
            
            # Only make prediction if we have enough data and at intervals
            prediction_counter += 1
            
            # Need full window before making predictions (matches training)
            if len(sequence_buffer) >= WINDOW_SIZE and prediction_counter >= PREDICTION_INTERVAL:
                prediction_counter = 0  # Reset counter
                
                # Convert buffer to numpy array
                sequence_array = np.array(sequence_buffer)
                
                # Create aggregated features
                seq_features_dict = create_sequence_features(sequence_array, raw_feature_names)
                
                # Convert to ordered array matching training feature order
                seq_features_array = np.array([seq_features_dict[feat] for feat in expected_features])
                
                # Reshape for prediction
                seq_features_array = seq_features_array.reshape(1, -1)
                
                # Scale features
                seq_features_scaled = scaler.transform(seq_features_array)
                
                # Predict with probabilities
                probabilities = model.predict_proba(seq_features_scaled)[0]
                class_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[class_idx])
                
                # Update cache
                last_prediction = class_idx
                last_confidence = confidence
            else:
                # Use cached prediction
                class_idx = last_prediction
                confidence = 0
            
            # Write output in single operation: count + outputs
            mm.seek(flags_size + num_inputs * 4)
            mm.write(struct.pack(output_format, 2.0, float(class_idx), float(confidence)))
            
            # Set flags
            mm.seek(0)
            mm.write(struct.pack(flags_format, False, True, False))  # cpp_wrote=False, python_wrote=True
            
    mm.close()
    shm.close_fd()
    print("Sequence-based perturbation classifier shutdown", file=sys.stderr)

if __name__ == "__main__":
    main()
