from kedro.pipeline import Pipeline, node
from .nodes import missing_column_count,drop_empty_features,barplot,remove_rating_1,\
univariate_analysis,numerical_feature_analysis,univariate_barplots,diff_month,numerical_feature_processing

def create_pipeline(**kwargs):
    return Pipeline(
            [node(func=missing_column_count, inputs="clean_data",outputs="missing_count",name="missing_feature_count"),
             node(func=drop_empty_features, inputs="clean_data",outputs="clean_data_2", name="drop_empty_features"),
             node(func=barplot, inputs="clean_data_2",outputs="target_plot",name="barplot"),
	     node(func=remove_rating_1, inputs="clean_data_2",outputs="clean_data_3",name="remove_rating_1"),
             node(func=univariate_analysis, inputs="clean_data_3",outputs=["fig1","fig2","fig3","fig4","fig5","fig6","fig7","fig8","fig9"],name="univariate_analysis"),
             node(func=numerical_feature_processing,inputs="clean_data_3",outputs=["data_after_eda","fig10","fig11"],name="numerical_feature_processing"),
            ])

