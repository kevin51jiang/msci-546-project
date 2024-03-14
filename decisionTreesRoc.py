import utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import importlib

importlib.reload(utils)

res = utils.train_and_report(
    "Decision Tree ROC",
    DecisionTreeClassifier(),
    {
        "model__criterion": ["gini", "entropy"],
        "model__splitter": ["best", "random"],
        "model__max_depth": [3, 5, 7, 9, 11, 13],
        "model__min_samples_split": [2, 3, 4],
    },
)

print(res)
print("Done!")
