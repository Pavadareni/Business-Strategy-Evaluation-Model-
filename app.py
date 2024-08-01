import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV


# 1. Model Training and Evaluation
@st.cache
def train_model():
    # Load dataset
    data = pd.read_csv("dataset.csv")

    # Example preprocessing
    data["user_growth_rate"] = (data["new_users"] - data["users_left"]) / data[
        "existing_users_before"
    ]

    # Define features and target
    X = data[
        [
            "new_users",
            "users_left",
            "existing_users_before",
            "existing_users_after",
            "user_growth_rate",
        ]
    ]
    y = data["strategy_effectiveness"].map({"positive": 1, "negative": 0, "average": 2})

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define parameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Initialize RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Define the scoring metric as recall
    scorer = make_scorer(recall_score, average="macro")

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring=scorer
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    # st.write("### Classification Report:")
    # st.text(
    #    classification_report(
    #        y_test, y_pred, target_names=["negative", "positive", "average"]
    #    )
    # )
    recall = recall_score(y_test, y_pred, average="macro")
    # st.write(f"### Recall Score: {recall:.4f}")

    return best_model


# 2. Prediction Function
def make_predictions(model, uploaded_file):
    try:
        if uploaded_file is None:
            st.error("No file uploaded.")
            return None, None

        # Read the CSV file into a DataFrame
        new_data = pd.read_csv(uploaded_file)

        # Check for missing values and handle them
        if new_data.isnull().values.any():
            st.error(
                "Uploaded file contains missing values. Please clean the data and try again."
            )
            return None, None

        # Example preprocessing (ensure consistency with training data preprocessing)
        new_data["user_growth_rate"] = (
            new_data["new_users"] - new_data["users_left"]
        ) / new_data["existing_users_before"]

        # Define features
        required_columns = [
            "new_users",
            "users_left",
            "existing_users_before",
            "existing_users_after",
            "user_growth_rate",
        ]

        # Ensure all required columns are present
        if not all(column in new_data.columns for column in required_columns):
            st.error(
                "Uploaded file is missing required columns. Please check your CSV file."
            )
            return None, None

        # Prepare data for prediction
        X_new = new_data[required_columns]

        # Make predictions
        predictions = model.predict(X_new)

        # Map predictions back to labels
        label_mapping = {1: "positive", 0: "negative", 2: "average"}
        predicted_labels = [label_mapping[pred] for pred in predictions]

        # Add predictions to the DataFrame
        new_data["predicted_strategy_effectiveness"] = predicted_labels

        # Aggregate results to give a single output
        positive_count = (
            new_data["predicted_strategy_effectiveness"] == "positive"
        ).sum()
        negative_count = (
            new_data["predicted_strategy_effectiveness"] == "negative"
        ).sum()
        average_count = (
            new_data["predicted_strategy_effectiveness"] == "average"
        ).sum()

        # Determine the overall strategy effectiveness
        if positive_count > negative_count and positive_count > average_count:
            overall_outcome = "Overall Strategy is positive and can be used for the next upcoming years."
        elif negative_count > positive_count and negative_count > average_count:
            overall_outcome = "Overall Strategy isn't working."
        else:
            overall_outcome = "Overall Strategy is working but upgradation is required."

        return new_data, overall_outcome

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None


# 3. Streamlit App Layout and Main Function
def main():
    st.title("Strategy Effectiveness Prediction")

    # Train model
    model = train_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data, outcome = make_predictions(model, uploaded_file)
        if data is not None:
            st.write("### Overall Outcome")
            st.write(outcome)
            st.write("### Uploaded Data with Predictions")
            st.write(data)


if __name__ == "__main__":
    main()
