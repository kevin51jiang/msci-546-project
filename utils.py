import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def get_data(split_train: bool = False):
    df = pd.read_csv('data/train.csv')

    # Split the data into training and testing data
    # X = df.drop(['Exited', 'Surname', 'CustomerId', 'Gender', 'Geography'], axis=1)
    df['SurnameLen'] = df['Surname'].apply(lambda x: len(x))
    X_raw = df.drop(['Exited', 'Surname', 'CustomerId'], axis=1)

    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=42)

    if split_train:
        # Split X_train and y_train into training and validation data
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train,
                                                          random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test, y
    else:
        return X_train, X_test, y_train, y_test, y


def report_data(model_name: str, y, y_test, y_pred, y_pred_proba, classes):
    def plot_overall_and_confusion():
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)
        roc_auc = roc_auc_score(y_test, y_pred)
        print("ROC AUC: ", roc_auc)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        pr_auc = auc(fpr, tpr)
        print("Precision/Recall AUC: ", pr_auc)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion matrix: ", conf_matrix)
        conf_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                           display_labels=classes)
        conf_plot.plot()
        plt.savefig(f'report/image/confusion_matrix_{model_name}.png')
        plt.show()

        # Save the confusion matrix to a file
        df = pd.DataFrame(conf_matrix)
        df.to_csv(f'report/text/confusion_matrix_{model_name}.csv', index=False)
        # save the scores to a file
        scores = pd.DataFrame({'accuracy': [accuracy], 'roc_auc': [roc_auc], 'pr_auc': [pr_auc]})
        scores.to_csv(f'report/text/scores_{model_name}.csv', index=False)
        plt.close()

    def plot_precision_recall():
        # Plot precision recall curves
        from sklearn.metrics import precision_recall_curve

        # retrieve just the probabilities for the positive class
        pos_probs = y_pred_proba[:, 1]
        # calculate the no skill line as the proportion of the positive class
        no_skill = len(y[y == 1]) / len(y)
        # plot the no skill precision-recall curve
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        # calculate model precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, pos_probs)
        # plot the model precision-recall curve
        plt.plot(recall, precision, marker='.', label=model_name)
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # Set y to be between 0 and 1
        plt.ylim([0.0, 1.05])
        # show the legend
        plt.legend()
        # show the plot

        plt.savefig(f'report/image/precision_recall_{model_name}.png')
        # Save the x,y values to a file
        df = pd.DataFrame({'precision': precision, 'recall': recall})
        df.to_csv(f'report/text/precision_recall_{model_name}.csv', index=False)
        plt.close()

    def plot_roc_curve():
        from sklearn.metrics import roc_curve

        # Source: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
        # retrieve just the probabilities for the positive class
        pos_probs = y_pred_proba[:, 1]
        # plot no skill roc curve
        plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        # calculate roc curve for model
        fpr, tpr, _ = roc_curve(y_test, pos_probs)
        # plot model roc curve
        #  marker='.',
        plt.plot(fpr, tpr, label=model_name)
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()

        # Save the plot to a file
        plt.savefig(f'report/image/roc_curve_{model_name}.png')
        # Save the x,y values to a file

        df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        df.to_csv(f'report/text/roc_curve_{model_name}.csv', index=False)
        plt.close()

    plot_overall_and_confusion()
    plot_precision_recall()
    plot_roc_curve()

    print("Done saving data!")
