import utils
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import importlib

importlib.reload(utils)

res = utils.train_and_report("Logistic Regression", LogisticRegression(), {})

print(res)
print("Done!")
