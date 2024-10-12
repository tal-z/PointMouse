import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_and_split_data(infile="your_data_file.csv"):
    # Load the data
    data = pd.read_csv(infile)

    # Separate features and labels
    X = data.iloc[:, 1:].values  # All columns except the first one
    y = data.iloc[:, 0].values  # The first column (labels)

    # Print unique labels before encoding
    print("Unique labels before encoding:", np.unique(y))

    # Encode the labels if they are categorical
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Print unique labels after encoding
    print("Unique labels after encoding:", np.unique(y_encoded))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, label_encoder


def hyperparameter_tuning_for_stacking(X_train, y_train):
    # Define base models for stacking
    base_estimators = [
        ("rf", RandomForestClassifier(random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
        ("xgb", XGBClassifier(eval_metric="mlogloss", random_state=42)),
    ]

    # Define meta-model
    meta_model = LogisticRegression()

    # Create StackingClassifier
    stacking_clf = StackingClassifier(
        estimators=base_estimators, final_estimator=meta_model, cv=5
    )

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [None, 10, 20],
        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["linear", "rbf"],
        "xgb__n_estimators": [100, 200, 300],
        "xgb__learning_rate": [0.01, 0.1, 0.2],
        "xgb__max_depth": [3, 4, 5],
        "final_estimator__C": [
            0.1,
            1,
            10,
        ],  # Hyperparameter for meta-model LogisticRegression
    }

    # Perform RandomizedSearchCV to tune hyperparameters
    randomized_search = RandomizedSearchCV(
        stacking_clf,
        param_distributions=param_grid,
        n_iter=50,
        cv=3,
        scoring="accuracy",
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    randomized_search.fit(X_train, y_train)

    print("Best Parameters found for Stacking: ", randomized_search.best_params_)

    return randomized_search.best_estimator_


def test_model(model, testing_data, label_encoder):
    X_test, y_test = testing_data
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Print classification report
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


def save_model_to_pkl(model, outfile="trained_model.pkl"):
    joblib.dump(model, outfile)


if __name__ == "__main__":
    print("Loading and splitting data...")
    # Load and split data
    X_train, X_test, y_train, y_test, label_encoder = load_and_split_data(
        "right_hand_landmarks_labeled_20241012112300.csv"
    )

    # Tune hyperparameters and train stacked model
    print("Tuning hyperparameters for stacked model...")
    best_stacked_model = hyperparameter_tuning_for_stacking(X_train, y_train)

    # Test the tuned stacked model
    print("Testing tuned stacked model...")
    test_model(best_stacked_model, (X_test, y_test), label_encoder)

    # Save the tuned stacked model
    print("Saving tuned stacked model...")
    save_model_to_pkl(best_stacked_model, outfile="tuned_stacked_model_exp.pkl")
    print("Done!")
