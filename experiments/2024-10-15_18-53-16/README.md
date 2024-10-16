

# EXPERIMENT 2024-10-15_18-53-16

# Training Data Notes
Three gestures reflected in data:
- 1 finger: left click
- 2 fingers: right click
- Any resting hand position (fist, open hand): no click

Removed some features: z-coordinates are removed, to flatten the comparison space to 2D. It helps a lot.

# Model Training Notes
- Stack of models with RandomSearchCV hyperparameter tuning.
    
# Application Notes
- z-coords are screened out in `extract_features` to match number of features in training data
    