import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import importlib

importlib.reload(utils)

res = utils.train_and_report(
    "Random Forest ROC",
    RandomForestClassifier(),
    {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
)

print(res)
print("Done!")
