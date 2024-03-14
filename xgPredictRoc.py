import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import importlib
import xgboost

importlib.reload(utils)

res = utils.train_and_report_roc(
    "XGBoost ROC",
    xgboost.XGBClassifier(),
    {
        "model__learning_rate": [0.01, 0.1, 0.2, 0.5],
        "model__max_depth": [3, 5, 6, 8, 9],
        "model__sampling_method": ["uniform", "gradient_based"],
        "model__tree_method": ["hist", "approx", "exact"]
    },
)

print(res)
print("Done!")
