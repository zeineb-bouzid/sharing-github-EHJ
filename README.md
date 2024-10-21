# Code description

This repository contains the codes necessary to generate the results described in our paper "ECG-based machine learning for risk stratification of patients with suspected acute coronary syndrome".

To generate the Sankey diagram (Figure 4 B), please call the corresponding function (defined in main.py) as follows:
plot_Shapley_explainability(clf=rsf, df_test=X_test, cols=dfX.columns, set_name='Test', nf=20, masker_data=X_train)

To generate the feature importance ranking (Figure 5), please call the corresponding function (defined in main.py) as follows:
plot_Sankey_diagram(clusters_HEART, clusters, y_test, dpgmm)
