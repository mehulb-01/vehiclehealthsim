# Vehicle Health Monitor

An AI/ML predictive-maintenance dashboard featuring a realistic car-cluster-style interface. 9 vehicle sensors feed a trained logistic-regression model that estimates failure probability and vehicle health score, displayed on a professional automotive dashboard with self-simulating gauges, warning lights, and manual control options.

## Features

### Realistic Car Dashboard Interface
- **Professional Gauge Design**: Black background with red backlighting and white markings, matching real automotive instrument clusters
- **Dual Main Gauges**: Vehicle health (0-100) and Failure probability (0-100%) with needle movement from 0° to 180°
- **Warning Light System**: Parking brake, general warning, engine, and oil pressure indicators that activate based on sensor conditions
- **Mini Sensor Dials**: 9 individual sensor gauges with arc displays and manual slider controls
- **Self-Simulating Mode**: Automatic sensor value changes over time to demonstrate health deterioration patterns
- **Manual Control**: Drag any slider to override simulation and manually test sensor scenarios

### Monitoring Capabilities
- **9 Vehicle Sensors**: Engine temperature, RPM, oil pressure, vibration, battery voltage, coolant temperature, fuel consumption, vehicle speed, and operating hours
- **Real-time Risk Assessment**: Each sensor monitored against safe operating bands with visual color coding (green/yellow/red)
- **Failure Probability Prediction**: Logistic regression model trained on 12,000 simulated duty cycles
- **Health Score Calculation**: Average risk score inverted to show overall vehicle condition
- **Contributing Factors**: Real-time display of sensors most impacting failure probability

## Files
```
dashboard.html      The main dashboard interface. Open directly in any browser — no
                     server or install needed. All logic (gauges, simulation,
                     the trained model's math) runs client-side in plain JS.
train_model.py       Reproduces the synthetic dataset and the trained
                     logistic regression that dashboard.html uses.
model.json           The trained model's weights, sensor ranges and
                     evaluation metrics (output of train_model.py).
requirements.txt     numpy + scikit-learn, for train_model.py only.
README.md            This file - comprehensive documentation and usage guide.
```

## Getting Started

### Run the Dashboard
Simply open `dashboard.html` in a web browser (double-click it, or use `File > Open`).

The dashboard starts in "live" mode with automatic sensor simulation:
- Watch the gauges respond to simulated sensor changes
- See warning lights activate based on vehicle conditions
- Monitor health score and failure probability in real-time

### Manual Control
- **Drag any slider** to take manual control of that specific sensor
- **Pause/Resume button** to stop/start the automatic simulation
- **Reset button** to return all sensors to their baseline healthy values

### Understanding the Display
- **Green zone**: Sensor values within safe operating range
- **Yellow zone**: Sensor values approaching dangerous levels  
- **Red zone**: Sensor values outside safe operating range
- **Warning lights**: Activate when specific conditions are met (e.g., engine overheating, low oil pressure)

## How the Model Works

### Risk Score Calculation
Each sensor value is converted to a 0–1 **risk score** against its own safe operating band:
- **0**: Inside the safe band (no risk)
- **Increasing**: The further outside the safe band the reading is
- **Maximum 1.4**: Extremes beyond danger thresholds

### Failure Probability
A logistic regression — trained on 12,000 simulated duty cycles — combines the 8 risk scores (vehicle speed is shown for context but excluded, since speed alone isn't a wear indicator) into a failure probability:

```
logit = intercept + Σ (weight_i × risk_i)
failure_probability = sigmoid(logit)
```

### Vehicle Health Score
The **vehicle health score** is a separate, simpler calculation: a plain average of the same 8 risk scores, inverted (100 = no risk at all). Because it isn't run through the logistic model, health score and failure probability can disagree — by design, so the model can flag a developing problem before the vehicle's overall condition visibly drops.

### Contributing Factors
The dashboard displays the **major contributing factors** — sensors with the highest `weight × risk`, i.e. the ones actually pushing the current failure probability up. This shows what the model itself is weighting, not a separate explanation layer.

## Model Retraining

### Requirements
```bash
pip install -r requirements.txt
```

### Training
```bash
python train_model.py
```

This regenerates `model.json` and prints accuracy / precision / recall / F1 / ROC-AUC for both a logistic regression and a random forest baseline (the random forest is shown for comparison only — the dashboard uses the logistic regression because its weights are directly interpretable, which matters for a system meant to explain *why* it's flagging risk).

### Updating the Dashboard
If you retrain and get different weights, copy the new `weights` and `intercept` values from `model.json` into `MODEL` near the top of the `<script>` block in `dashboard.html`.

## Technical Implementation

### Frontend Architecture
- **Pure HTML/CSS/JavaScript**: No frameworks or dependencies required
- **SVG-based Gauges**: Scalable vector graphics for crisp rendering at any size
- **Responsive Design**: Adapts to different screen sizes with CSS Grid
- **Real-time Updates**: 900ms simulation interval with smooth CSS transitions

### Model Integration
- **Client-side Computation**: All risk calculations run in the browser
- **Pre-trained Weights**: Model coefficients embedded directly in the JavaScript
- **Instant Response**: No server calls or API dependencies

### Visual Design
- **Automotive Theme**: Inspired by real car instrument clusters
- **Color Coding**: Green (healthy), Yellow (warning), Red (danger)
- **Smooth Animations**: CSS transitions for needle movement and gauge updates
- **Warning System**: Dynamic light activation based on sensor thresholds

## Limitations and Honesty

### Data Quality
- The dataset is **synthetic**, generated from domain-reasonable sensor ranges and degradation patterns, not real vehicle telemetry. It's built to demonstrate the full pipeline (data → model → explainable score → UI), not to be production-accurate.

### Model Performance
- Recall is modest (0.28 at the default 0.5 threshold) because failure events are intentionally rare in the simulated data, same as in a real fleet. A production system would tune the decision threshold and collect more failure examples rather than relying on accuracy alone (accuracy is 93% but is a misleading headline metric here, precisely because failures are rare).

### Sensor Specifications
- The 9 sensors and their safe/danger ranges are reasonable general assumptions for a passenger vehicle engine, not vehicle-specific specifications.

### Simulation Accuracy
- The self-simulating mode uses random variations and smoothing algorithms to demonstrate dynamic behavior, but does not represent actual vehicle driving patterns or real-world sensor relationships.

## Use Cases

### Educational
- Demonstrate predictive maintenance concepts
- Show logistic regression in action
- Explain sensor monitoring and risk assessment

### Prototyping
- Test UI/UX designs for vehicle monitoring systems
- Evaluate different sensor configurations
- Validate warning system behavior

### Research
- Experiment with different model weights
- Compare various sensor combinations
- Study failure prediction accuracy

## Contributing

This is a demonstration project for educational and prototyping purposes. Feel free to:
- Fork the repository
- Experiment with different sensor configurations
- Modify the visual design
- Retrain the model with different parameters
- Submit issues or suggestions

## License

This project is provided as-is for educational and demonstration purposes.

## Credits

Developed as a vehicle health monitoring simulation system demonstrating AI/ML applications in predictive maintenance.
