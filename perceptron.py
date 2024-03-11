from sklearn.decomposition import PCA
import utils
from sklearn.neural_network import MLPClassifier

res = utils.train_and_report(
    "Multi Layer Perceptron",
    MLPClassifier(),
    {
        "model__hidden_layer_sizes": [
            # Given the same 128 neurons, what's the "most effective" way we can make use of it?
            (128,),
            (64, 64),
            (32, 32, 32, 32),
            (16, 16, 16, 16, 16, 16, 16, 16),
            (8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8),
            (8, 16, 16, 64, 16, 8),
            (32, 16, 8, 8, 8, 8 ,8, 8, 8, 8, 16, 32,),
            (8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 32, 32),
            (32, 32, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8),
        ],
        "model__activation": ["logistic", "relu"],
        "model__solver": ["sgd", "adam"],
        "model__alpha": [0.0001, 0.001, 0.01],
        "model__learning_rate": ["constant", "invscaling", "adaptive"],
        "model__max_iter": [1000],
        # This is already gonna take a lot of processing time, let's stick with what we know is better
        "reduce_dim": [PCA()],
        "reduce_dim__n_components": ["mle"],
    },
)

print(res)
print("Done!")
