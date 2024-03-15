# msci-546-project


Team 1

Workflow to add Model (after EDA):
1. copy `decisionTrees.py` to `<your_model_name>.py`
2. Add hyperparameters that you want to test in `<your_model_name>.py`
3. modify `utils.py` (train & report functions) if needed
4. Run `<your_model_name>.py`
5. Results:
   1. Images stored in `/report/image` folder
   7. Models stored in `/models` folder
   8. All text results are stored in `/report/text` folder such as:
      1. `best_params_<your_model_name>.csv` - best hyperparameters for the best model
      10. `confusion_matrix_<your_model_name>.csv` - confusion matrix for the best model
      11. `cv_results_<your_model_name>.csv` - results from Grid CV search for all the different hyperparameters. Shows all models that have been trained
      12. `precision_recall_<your_model_name>.csv` - precision and recall raw graph x/y values
      13. `roc_curve_<your_model_name>.csv` - ROC curve raw graph x/y values
      14. `scores_<your_model_name>.csv` - scores for the best model (accuracy, roc auc, pr auc)
