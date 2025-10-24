# ML Perturbation Classifier Integration

## Overview
The ML-based perturbation classifier has been integrated into the ForceCheck system via PythonScriptCaller. It uses a trained Random Forest model to classify perturbations in real-time.

## Architecture

```
Servos.PRESENT_CURRENT[0:2] ──┐
                               ├──> PerturbationClassifier ──> ForceCheck.PerturbationClassification
CurrentPrediction.Output ──────┘
```

## Data Flow

### Inputs to PerturbationClassifier (4 floats):
- `Input[0]`: Tilt motor current (from `Servos.PRESENT_CURRENT[0]`)
- `Input[1]`: Pan motor current (from `Servos.PRESENT_CURRENT[1]`)
- `Input[2]`: Tilt motor prediction (from `CurrentPrediction.Output[0]`)
- `Input[3]`: Pan motor prediction (from `CurrentPrediction.Output[1]`)

### Outputs from PerturbationClassifier (2 floats):
- `Output[0]`: Class index (0=none, 1=obstacle, 2=push, 3=sustained)
- `Output[1]`: Confidence (0.0-1.0)

## Files Modified

### 1. `/Users/pierre/ikaros/UserData/ForceRegulation_test.ikg`
Added PerturbationClassifier module and connections:

```xml
<!-- ML Perturbation Classifier -->
<module NumberInputs="4" 
        NumberOutputs="2" 
        ScriptPath="/Users/pierre/ikaros/Source/Modules/RobotModules/CompliantRobot/ForceCheck/perturbation_classifier.py" 
        SharedMemoryName="perturbation_classifier_shm" 
        VenvPath="/Users/pierre/ikaros/Source/Modules/RobotModules/CompliantRobot/TestMapping/.tensorflow_venv/bin/python3.12" 
        Timeout="100" 
        _x="200" 
        _y="816" 
        class="PythonScriptCaller" 
        log_level="7" 
        name="PerturbationClassifier"/>

<!-- Connections -->
<connection delay="1" source="Servos.PRESENT_CURRENT[0:2]" target="PerturbationClassifier.Input[0:2]"/>
<connection delay="1" source="CurrentPrediction.Output" target="PerturbationClassifier.Input[2:4]"/>
<connection delay="1" source="PerturbationClassifier.Output" target="ForceCheck.PerturbationClassification"/>
```

### 2. `/Users/pierre/ikaros/Source/Modules/RobotModules/CompliantRobot/ForceCheck/ForceCheck.ikc`
Added input for receiving classification results:

```xml
<input name="PerturbationClassification" optional="true" 
       description="The perturbation classifier output from PythonScriptCaller" />
```

### 3. Python Script: `perturbation_classifier.py`
Location: `/Users/pierre/ikaros/Source/Modules/RobotModules/CompliantRobot/ForceCheck/perturbation_classifier.py`
- Loads trained Random Forest model
- Communicates via shared memory
- Returns classification results

### 4. Model Files
- `/Users/pierre/ikaros/Source/Modules/RobotModules/CompliantRobot/ForceCheck/models/perturbation_classifier_rf.pkl`
- `/Users/pierre/ikaros/Source/Modules/RobotModules/CompliantRobot/ForceCheck/models/feature_scaler.pkl`

## Integration in ForceCheck.cc

### 1. Add Member Variable
```cpp
class ForceCheck: public Module {
    // ... existing members ...
    
    matrix perturbation_classification;   // Input from ML classifier
    bool use_ml_classifier;               // Flag to check if ML is available
};
```

### 2. In Init()
```cpp
void Init() {
    // ... existing Init code ...
    
    // Bind ML classifier input
    Bind(perturbation_classification, "PerturbationClassification");
    
    // Check if ML classifier is connected
    use_ml_classifier = perturbation_classification.connected();
    
    if (use_ml_classifier) {
        Debug("ML Classifier ENABLED");
    } else {
        Debug("ML Classifier DISABLED - using rule-based system only");
    }
}
```

### 3. In Tick() or ClassifyProfiles()
```cpp
void ClassifyProfilesWithML(std::vector<ErrorProfile>& profiles) {
    // Read ML classification result if available
    std::string ml_class_type = "none";
    double ml_confidence = 0.0;
    
    if (use_ml_classifier && perturbation_classification.connected() && 
        perturbation_classification.size() >= 2) {
        
        int class_idx = (int)perturbation_classification(0);
        ml_confidence = perturbation_classification(1);
        
        // Map class index to string
        switch (class_idx) {
            case 0: ml_class_type = "none"; break;
            case 1: ml_class_type = "obstacle"; break;
            case 2: ml_class_type = "push"; break;
            case 3: ml_class_type = "sustained_force"; break;
            default: ml_class_type = "unknown"; break;
        }
        
        Debug("ML Classification: " + ml_class_type + 
              " (confidence: " + std::to_string(ml_confidence) + ")");
    }
    
    // Apply ML classification to BOTH motors if confident
    if (ml_confidence > 0.70 && profiles.size() >= 2) {
        profiles[0].pattern_type = ml_class_type;
        profiles[0].confidence = ml_confidence;
        profiles[1].pattern_type = ml_class_type;
        profiles[1].confidence = ml_confidence;
        
        std::cout << "Using ML classification: " << ml_class_type 
                  << " (confidence: " << ml_confidence << ")" << std::endl;
        return;  // Skip rule-based classification
    }
    
    // Fall back to rule-based classification if ML not confident
    ClassifyProfiles(profiles);  // Your existing rule-based function
}
```

## Model Performance

- **Overall Accuracy**: 80.2%
- **Push Detection**: 94.9% (excellent)
- **Sustained Force**: 81.6% (good)
- **Obstacle**: 74.1% (moderate)
- **None**: 73.8% (moderate)

### Feature Importance
1. Tilt error: 37.3%
2. Tilt current: 27.6%
3. Pan current: 22.2%
4. Pan error: 13.0%

## Usage

The ML classifier runs automatically when:
1. The `.ikg` file includes the PerturbationClassifier module
2. The Python script and model files are in place
3. The virtual environment is available

If any component is missing, ForceCheck automatically falls back to the rule-based system.

## Configuration Parameters

### PerturbationClassifier Module
- `NumberInputs`: 4 (fixed)
- `NumberOutputs`: 2 (fixed)
- `Timeout`: 100 ms (adjustable if needed)
- `SharedMemoryName`: Must be unique per instance

### ForceCheck
- `MinConfidenceForStateSwitch`: Threshold for trusting ML classification (default: 0.8)
  - Lower values (0.6-0.7): More responsive, but may have more false positives
  - Higher values (0.8-0.9): More conservative, but higher accuracy

## Troubleshooting

### ML Classifier Not Working
1. Check that Python script exists: `perturbation_classifier.py`
2. Check that model files exist in `models/` directory
3. Check virtual environment path in `.ikg` file
4. Check log output for PythonScriptCaller errors

### Falling Back to Rule-Based
If ML classifier is not available or confidence is low (<0.70), the system automatically uses the existing rule-based classification. This ensures the robot always has a fallback mechanism.

## Future Improvements

1. **Add temporal features**: Window statistics could improve accuracy to 85-88%
2. **Per-motor classification**: Currently classifies both motors together
3. **Dynamic confidence threshold**: Adjust based on context
4. **Online learning**: Update model based on user corrections
