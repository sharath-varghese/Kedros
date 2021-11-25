
from kedro.pipeline import Pipeline, node
from .nodes import plot_confusion_matrix,defining_model,training,rf_training,model_tuning

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(func=model_tuning,
                 inputs=["X_tr","y_train","X_cr","y_cv","X_te","y_test"],
                 outputs=["plot_knn","confusion_knn","precision_knn","recall_knn","model_knn",\
                          "plot_lr","confusion_lr","precision_lr","recall_lr","model_lr",\
                          "plot_svm","confusion_svm","precision_svm","recall_svm","model_svm",\
                          "confusion_rf","precision_rf","recall_rf","model_rf"]),
        ])
