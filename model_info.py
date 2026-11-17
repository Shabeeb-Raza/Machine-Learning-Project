MODEL_INFO = {
    "SVM": {
        "summary": "Finds a separating boundary by maximising the margin between classes.",
        "strengths": [
            "Strong on small and medium datasets with clear margins.",
            "Kernel options capture non-linear boundaries.",
        ],
        "weaknesses": [
            "Needs scaled numeric features.",
            "Training can become slow as rows grow.",
            "Probabilities are not always well calibrated.",
        ],
        "use_cases": "Text-style features, small tabular datasets, non-linear boundaries.",
    },
    "Logistic Regression": {
        "summary": "A linear classifier that estimates class probabilities from a weighted feature score.",
        "strengths": [
            "Fast, stable, and easy to interpret.",
            "Outputs useful probabilities.",
            "Works well as a first baseline.",
        ],
        "weaknesses": [
            "Cannot capture complex non-linear patterns without feature engineering.",
            "Sensitive to unscaled features when regularisation is used.",
        ],
        "use_cases": "Churn, credit risk, medical screening, and explainable scoring.",
    },
    "KNN Classifier": {
        "summary": "Predicts the class shared by the closest training rows.",
        "strengths": [
            "Simple and intuitive.",
            "Captures local patterns without a training phase.",
        ],
        "weaknesses": [
            "Needs scaled features.",
            "Prediction gets slower with more rows.",
            "Sensitive to irrelevant features and class imbalance.",
        ],
        "use_cases": "Small recommendation or similarity problems.",
    },
    "Decision Tree": {
        "summary": "Learns a sequence of if/else splits that separate the target classes.",
        "strengths": [
            "Easy to explain and visualise.",
            "Handles mixed feature types with little preparation.",
        ],
        "weaknesses": [
            "Can overfit noisy data.",
            "Small data changes can produce a very different tree.",
        ],
        "use_cases": "Rule discovery and explainable screening.",
    },
    "Random Forest": {
        "summary": "Averages many decision trees to produce a more stable classifier.",
        "strengths": [
            "Strong general-purpose accuracy.",
            "Handles non-linear relationships and interactions.",
            "Provides built-in feature importance.",
        ],
        "weaknesses": [
            "Less interpretable than a single tree.",
            "Can be slower and heavier on large grids.",
        ],
        "use_cases": "General tabular classification and feature ranking.",
    },
    "SVR": {
        "summary": "Fits a regression line inside an error-tolerant tube using support vectors.",
        "strengths": [
            "Flexible non-linear regression through kernels.",
            "Works well on smaller datasets.",
        ],
        "weaknesses": [
            "Needs scaled numeric features.",
            "Can be slow on larger datasets.",
            "Hyperparameters need careful tuning.",
        ],
        "use_cases": "Small regression problems with non-linear structure.",
    },
    "Ridge Regression": {
        "summary": "A linear regression model that shrinks coefficients to reduce overfitting.",
        "strengths": [
            "Fast, stable, and interpretable.",
            "Handles correlated numeric features better than plain least squares.",
        ],
        "weaknesses": [
            "Assumes a mostly linear relationship.",
            "Needs sensible scaling when features have very different ranges.",
        ],
        "use_cases": "Price, demand, and continuous scoring baselines.",
    },
    "KNN Regressor": {
        "summary": "Predicts a value by averaging the closest training rows.",
        "strengths": [
            "Simple and local.",
            "Captures non-linear patterns without a fitted equation.",
        ],
        "weaknesses": [
            "Needs scaled features.",
            "Sensitive to irrelevant features and outliers.",
            "Prediction cost grows with dataset size.",
        ],
        "use_cases": "Small similarity-based regression problems.",
    },
    "Random Forest Regressor": {
        "summary": "Averages many regression trees for a stable continuous prediction.",
        "strengths": [
            "Strong general-purpose regression accuracy.",
            "Captures interactions and non-linear relationships.",
            "Provides built-in feature importance.",
        ],
        "weaknesses": [
            "Can be slower than linear models.",
            "Does not extrapolate beyond the training target range.",
        ],
        "use_cases": "General tabular regression and driver analysis.",
    },
}


def info_for(model_name):
    return MODEL_INFO.get(
        model_name,
        {
            "summary": "A scikit-learn estimator from the model registry.",
            "strengths": ["Configured through the shared model registry."],
            "weaknesses": ["Check the model documentation for assumptions."],
            "use_cases": "General tabular prediction.",
        },
    )
