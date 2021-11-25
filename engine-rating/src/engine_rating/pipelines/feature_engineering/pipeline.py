from kedro.pipeline import Pipeline, node
from .nodes import missing_feature_cols,train_cv_test_split,fit_response_encoding,transfrom_response_encoding,\
    response_encoding,encoding_numerical_features,combining_features


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(func=missing_feature_cols,
                inputs="data_after_eda",
                outputs=["data_after_fe","feature_cols","fe_cols"],
                name="missing_feature_cols"),

            node(func=train_cv_test_split,
                inputs="data_after_fe",
                outputs=["X_train","y_train","X_cv","y_cv","X_test","y_test"],
                name="train_cv_test_split"),

            node(func=response_encoding,
                inputs=["feature_cols","X_train","X_cv","X_test","y_train","y_cv","y_test"],
                outputs="encoded_features",
                name="response_encoding"),

            node(func=encoding_numerical_features,
                inputs=["X_train","X_cv","X_test","y_train","y_cv","y_test"],
                outputs=["X_train_odo","X_cv_odo","X_test_odo","X_train_time","X_cv_time","X_test_time"],
                name="encoding_numerical_features"),

            node(func=combining_features,
                inputs=["encoded_features","X_train_odo","X_cv_odo","X_test_odo","X_train_time","X_cv_time","X_test_time","X_train","X_cv","X_test","fe_cols"],
                outputs=["X_tr","X_cr","X_te"],
                name="combining_features"),
        ])
