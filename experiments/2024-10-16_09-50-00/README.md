

# EXPERIMENT 2024-10-16_09-50-00
UNSUCCESSFUL

# Training Data Notes
Three gestures reflected in data:
- pinch: left click
- fist: right click
- Any other hand position (open hand, pointing): no click

Removed some features: z-coordinates are removed, to flatten the comparison space to 2D.

# Model Training Notes
- Stack of models with RandomSearchCV hyperparameter tuning.
    
# Application Notes
- z-coords are screened out in `extract_features` to match number of features in training data
- Does not perform better than version with all features (including Z)