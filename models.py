from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

CLASSIFICATION = "classification"
REGRESSION = "regression"

MODEL_REGISTRY = {
    "SVM": {
        "task": CLASSIFICATION,
        "class": SVC,
        "fixed_params": {},
        "hyperparameters": {
            "kernel": {
                "type": "select",
                "options": ["rbf", "linear", "poly", "sigmoid"],
                "default": "rbf",
                "help": "Kernel function used by the support vector machine.",
            },
            "C": {
                "type": "float",
                "min": 0.01,
                "max": 100.0,
                "step": 0.01,
                "default": 1.0,
                "help": "Regularisation strength. Smaller values mean stronger regularisation.",
            },
            "gamma": {
                "type": "select",
                "options": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                "default": "scale",
                "help": "Kernel coefficient. Use a preset or a fixed numeric value.",
            },
            "degree": {
                "type": "int",
                "min": 1,
                "max": 5,
                "step": 1,
                "default": 3,
                "help": "Polynomial degree. Only used when the kernel is poly.",
            },
        },
    },
    "Logistic Regression": {
        "task": CLASSIFICATION,
        "class": LogisticRegression,
        "fixed_params": {},
        "hyperparameters": {
            "C": {
                "type": "float",
                "min": 0.01,
                "max": 100.0,
                "step": 0.01,
                "default": 1.0,
                "help": "Inverse regularisation strength.",
            },
            "solver": {
                "type": "select",
                "options": ["lbfgs", "liblinear"],
                "default": "lbfgs",
                "help": "Optimisation algorithm used to fit the model.",
            },
            "max_iter": {
                "type": "int",
                "min": 100,
                "max": 3000,
                "step": 100,
                "default": 1000,
                "help": "Maximum solver iterations.",
            },
        },
    },
    "KNN Classifier": {
        "task": CLASSIFICATION,
        "class": KNeighborsClassifier,
        "fixed_params": {},
        "hyperparameters": {
            "n_neighbors": {
                "type": "int",
                "min": 1,
                "max": 30,
                "step": 1,
                "default": 5,
                "help": "Number of nearest neighbours used for a prediction.",
            },
            "weights": {
                "type": "select",
                "options": ["uniform", "distance"],
                "default": "uniform",
                "help": "Whether neighbours vote equally or by inverse distance.",
            },
            "metric": {
                "type": "select",
                "options": ["minkowski", "euclidean", "manhattan"],
                "default": "minkowski",
                "help": "Distance metric used to find neighbours.",
            },
        },
    },
    "Decision Tree": {
        "task": CLASSIFICATION,
        "class": DecisionTreeClassifier,
        "fixed_params": {},
        "hyperparameters": {
            "criterion": {
                "type": "select",
                "options": ["gini", "entropy"],
                "default": "gini",
                "help": "Function used to measure split quality.",
            },
            "max_depth": {
                "type": "int",
                "min": 1,
                "max": 30,
                "step": 1,
                "default": 5,
                "help": "Maximum depth of the tree.",
            },
            "min_samples_split": {
                "type": "int",
                "min": 2,
                "max": 20,
                "step": 1,
                "default": 2,
                "help": "Minimum number of samples required to split a node.",
            },
        },
    },
    "Random Forest": {
        "task": CLASSIFICATION,
        "class": RandomForestClassifier,
        "fixed_params": {},
        "hyperparameters": {
            "n_estimators": {
                "type": "int",
                "min": 10,
                "max": 300,
                "step": 10,
                "default": 100,
                "help": "Number of trees in the forest.",
            },
            "max_depth": {
                "type": "int",
                "min": 1,
                "max": 30,
                "step": 1,
                "default": 10,
                "help": "Maximum depth of each tree.",
            },
            "min_samples_split": {
                "type": "int",
                "min": 2,
                "max": 20,
                "step": 1,
                "default": 2,
                "help": "Minimum number of samples required to split a node.",
            },
            "max_features": {
                "type": "select",
                "options": ["sqrt", "log2"],
                "default": "sqrt",
                "help": "Number of features considered when looking for the best split.",
            },
        },
    },
    "SVR": {
        "task": REGRESSION,
        "class": SVR,
        "fixed_params": {},
        "hyperparameters": {
            "kernel": {
                "type": "select",
                "options": ["rbf", "linear", "poly", "sigmoid"],
                "default": "rbf",
                "help": "Kernel function used by the support vector regressor.",
            },
            "C": {
                "type": "float",
                "min": 0.01,
                "max": 100.0,
                "step": 0.01,
                "default": 1.0,
                "help": "Regularisation strength. Smaller values mean stronger regularisation.",
            },
            "gamma": {
                "type": "select",
                "options": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                "default": "scale",
                "help": "Kernel coefficient. Use a preset or a fixed numeric value.",
            },
            "degree": {
                "type": "int",
                "min": 1,
                "max": 5,
                "step": 1,
                "default": 3,
                "help": "Polynomial degree. Only used when the kernel is poly.",
            },
            "epsilon": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "default": 0.1,
                "help": "Width of the error-free tube around the regression line.",
            },
        },
    },
    "Ridge Regression": {
        "task": REGRESSION,
        "class": Ridge,
        "fixed_params": {},
        "hyperparameters": {
            "alpha": {
                "type": "float",
                "min": 0.01,
                "max": 100.0,
                "step": 0.01,
                "default": 1.0,
                "help": "Regularisation strength. Larger values shrink coefficients more.",
            },
            "solver": {
                "type": "select",
                "options": ["auto", "svd", "cholesky", "lsqr"],
                "default": "auto",
                "help": "Numerical solver used to fit the model.",
            },
        },
    },
    "KNN Regressor": {
        "task": REGRESSION,
        "class": KNeighborsRegressor,
        "fixed_params": {},
        "hyperparameters": {
            "n_neighbors": {
                "type": "int",
                "min": 1,
                "max": 30,
                "step": 1,
                "default": 5,
                "help": "Number of nearest neighbours used for a prediction.",
            },
            "weights": {
                "type": "select",
                "options": ["uniform", "distance"],
                "default": "uniform",
                "help": "Whether neighbours contribute equally or by inverse distance.",
            },
            "metric": {
                "type": "select",
                "options": ["minkowski", "euclidean", "manhattan"],
                "default": "minkowski",
                "help": "Distance metric used to find neighbours.",
            },
        },
    },
    "Random Forest Regressor": {
        "task": REGRESSION,
        "class": RandomForestRegressor,
        "fixed_params": {},
        "hyperparameters": {
            "n_estimators": {
                "type": "int",
                "min": 10,
                "max": 300,
                "step": 10,
                "default": 100,
                "help": "Number of trees in the forest.",
            },
            "max_depth": {
                "type": "int",
                "min": 1,
                "max": 30,
                "step": 1,
                "default": 10,
                "help": "Maximum depth of each tree.",
            },
            "min_samples_split": {
                "type": "int",
                "min": 2,
                "max": 20,
                "step": 1,
                "default": 2,
                "help": "Minimum number of samples required to split a node.",
            },
            "max_features": {
                "type": "select",
                "options": ["sqrt", "log2"],
                "default": "sqrt",
                "help": "Number of features considered when looking for the best split.",
            },
        },
    },
}


def models_for_task(task):
    return [name for name, entry in MODEL_REGISTRY.items() if entry["task"] == task]


def hyperparameters_for(model_name):
    return MODEL_REGISTRY[model_name]["hyperparameters"]


def search_candidates(spec):
    if spec["type"] == "select":
        return list(spec["options"])
    if spec["type"] in {"int", "float"}:
        return sorted({spec["min"], spec["default"], spec["max"]})
    return [spec["default"]]
