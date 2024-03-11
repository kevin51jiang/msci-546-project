import utils
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import importlib

importlib.reload(utils)

# To run this, temporarily modify utils.train_and_report to not do PCA
res = utils.train_and_report("Logistic Regression", LogisticRegression(), {})

print(res)
print("Done!")
