import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict

from sklearn.utils import shuffle

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer
# now you can import normally from sklearn.impute
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis, RandomSurvivalForest, ExtraSurvivalTrees
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score, as_concordance_index_ipcw_scorer
from sklearn.metrics import silhouette_score, mean_squared_error

import shap

from sksurv.compare import compare_survival

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

from sklearn.mixture import BayesianGaussianMixture

import plotly.graph_objects as go
import plotly.io as pio

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

import time
start_time = time.time()

list_hybrid_features_JAHA = ['rmsVar', 'pctSTNDPV', 'fpTinfl1Axis', 'st80_aVL', 'TrelAmp', 'PCA2', 'vat_V2', 'pctTNDPV', 'vat_V4', 'antConcaveAmp', 'J_PCAratio', 'vat_III', 'PR', 'TpTe', 'TMDpre', 'Age', 'txzQRSaxis', 'latConcaveAmp', 'rmsMin', 'pctRNDPV', 'JTc', 'T_PCAratio', 'STT_PCAratio', 'pctJNDPV', 'Tasym', 'PCA3', 'STNDPV', 'fpTaxis', 'fpTinfl1Mag', 'ramp_V4', 'vat_II', 'RNDPV', 'QRSTangle', 'TMDpost', 'ramp_III', 'spatialTaxisDev', 'mfpQRSaxis', 'ramp_aVL', 'qdur_aVF', 'TMD', 'TampInfl1', 'HR', 'TNDPV', 'TCRT', 'pcaTamp', 'PCA1', 'MIsz', 'JNDPV', 'mQRSTangle', 'TCRTangle', 'QRS_PCAratio', 'Tamp', 'QRSd', 'st80_V6', 'tamp_aVL', 'st80_V4', 'st80_V5', 'st80_aVF', 'tamp_aVR', 'tamp_V1', 'st80_V3', 'tamp_V6', 'tamp_I', 'st80_III', 'st80_V1', 'st80_I', 'st80_V2', 'tamp_V5', 'tamp_II', 'tamp_III', 'tamp_V4', 'tamp_V2', 'tamp_V3', 'tamp_aVF']

def parse_args():
    parser = argparse.ArgumentParser(
                        prog = 'Survival Analysis',
                        description = 'Predict mortality in chest-pain patients',
                        epilog = 'Please use a GPU to run this program as the grid search might need a lot of resources.')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for the cross-validation')
    parser.add_argument('-i', '--imputation_method', default='iterative', choices=['mean', 'knn', 'iterative'], help='Imputation method')
    parser.add_argument('-c', '--clf_name', default='EST', help='Classifier name')
    args = vars(parser.parse_args())
    return args

#DATA LOADING AND PREPROCESSING

def get_y(file_name='data/Main_Dataset_Outcomes.xlsx'):
    dfy = pd.ExcelFile(file_name).parse('Sheet1', skiprows=0)
    dfy = dfy[dfy['Exclusions']==0]
    dfy.set_index('ID', drop=True, inplace=True)
    dfy['Outcome_NDI_death'] = dfy['Outcome_NDI_death'].astype(bool)
    dfy = dfy.loc[:, ['Outcome_NDI_death', 'Outcome_NDI_Time']]
    return dfy

def get_X(file_name='data/Main_Dataset_Features.csv', reduce=True):
    dfX = pd.read_csv(file_name)
    dfX.set_index('ID', drop=True, inplace=True)
    dfX.drop([i for i in dfX.index if not i in dfy.index], axis = 0, inplace=True)
    #Dimensionality reduction (based on previous work)
    if reduce==True:
        listX_features = dfX.columns.values.tolist()
        for el in listX_features:
            if not el in list_hybrid_features_JAHA:
                dfX.drop(el, axis = 1, inplace=True)
    return dfX

def get_X_MEDIC(file_name='data/External_Validation_Features.csv', reduce=True):
    X_MEDIC = pd.read_csv(file_name)
    X_MEDIC.drop(X_MEDIC.index[X_MEDIC['first_primary'] == 0], inplace=True)
    X_MEDIC.dropna(subset=['first_primary'], inplace=True)
    #Dimensionality reduction (based on previous work)
    if reduce==True:
        MEDIC_features = X_MEDIC.columns.values.tolist()
        for el in MEDIC_features:
            if not el in list_hybrid_features_JAHA:
                X_MEDIC.drop(el, axis = 1, inplace=True)
    return X_MEDIC

def remove_missing_patients(dfX, dfy=None, percent_missingness=0.05):
    percent_missing_patients = pd.DataFrame(data= [dfX.iloc[i].isnull().sum()/dfX.shape[1] for i in range(len(dfX.index))],
                                        index= dfX.index)
    missing_patients_index = percent_missing_patients.index[percent_missing_patients[0]>percent_missingness].tolist()
    dfX.drop(missing_patients_index, axis = 0, inplace=True)
    if dfy is None:
        return dfX
    else:
        dfy.drop(missing_patients_index, axis = 0, inplace=True)
        return dfX, dfy
    
def clean_outliers(dfX):
    rules = [('TampInfl1', float('-inf'), 450),
             ('Tamp', float('-inf'), 1000),
             ('PCA1', float('-inf'), 4),
             ('TpTe', float('-inf'), 200),
             ('ramp_III', float('-inf'), 2000),
             ('ramp_aVL', float('-inf'), 2000),
             ('ramp_V4', float('-inf'), 2000),
             ('vat_II',	0, 100),
             ('vat_III', 0, 100), 
             ('vat_V2',	0, 100), 
             ('vat_V4',	0, 100),
             ('st80_I', -400, 700),
             ('st80_III', -400, 700),
             ('st80_aVL', -400, 700),
             ('st80_aVF', -400, 700),
             ('st80_V1', -400, 700),
             ('st80_V2', -400, 700),
             ('st80_V3', -400, 700),
             ('st80_V4', -400, 700),
             ('st80_V5', -400, 700),
             ('st80_V6', -400, 700),
             ('tamp_I', -1000, 1000), 
             ('tamp_II', -1000, 1000),
             ('tamp_III', -1000, 1000),
             ('tamp_aVR', -1000, 1000),
             ('tamp_aVL', -1000, 1000),
             ('tamp_aVF', -1000, 1000),
             ('tamp_V1', -1000, 1000),
             ('tamp_V2', -1000, 1000),
             ('tamp_V3', -1000, 1000),
             ('tamp_V4', -1000, 1000),
             ('tamp_V5', -1000, 1000),
             ('tamp_V6', -1000, 1000)
             ]
    for feature, minimum, maximum in rules:
        dfX.loc[dfX[feature]<minimum, feature] = minimum
        dfX.loc[dfX[feature]>maximum, feature] = maximum
    return dfX

def get_clusters_HEART(y_test, file_name='data/HEART_scores.xlsx'):
    dfy_HEART = pd.ExcelFile(file_name).parse('Sheet1', skiprows=0)
    dfy_HEART.set_index('ID', drop=True, inplace=True)
    #Reduce to test set
    y_HEART_clusters = dfy_HEART.loc[y_test.index, "HEART_classes"]
    y_HEART_score = dfy_HEART.loc[y_test.index, "HEART"]
    return y_HEART_clusters.values, y_HEART_score.values

def rearrange_colors(x, color_list = np.array(["green", "orange", "red"])):
    i = np.searchsorted(np.sort(x), x)
    return color_list[i]

def plot_results(X, Y_, title, y_test, dpgmm, set_name):
    color_list = rearrange_colors(dpgmm.means_.squeeze())
    label_list = rearrange_colors(dpgmm.means_.squeeze(), np.array(['Low risk', 'Moderate risk', 'High risk']))
    
    plt.figure(figsize=(8, 16))
    
    x = np.linspace(min(X), max(X), 1000)
    pdf = np.exp(dpgmm.score_samples(x.reshape(-1, 1)))
    pdf_clusters = dpgmm.predict_proba(x.reshape(-1, 1)) * pdf[:, np.newaxis]

    #PLOT 1
    plt.subplot(2, 1, 1)
    labels = [label_list[i]+' (n = '+str(sum(Y_==i))+')' for i in np.arange(3)]
    plt.hist([X[Y_==i] for i in np.arange(3)], 50, density=True, color=color_list, label=labels, stacked=True)
    plt.plot(x, pdf, '-k', linewidth=2)
    plt.plot(x, pdf_clusters, '--k', linewidth=2)
    plt.xlabel('Predicted score')
    plt.ylabel('Density')
    plt.legend(loc="best")
    plt.title(title)
    
    #PLOT 2
    
    ax = plt.subplot(2, 1, 2)
    kmf_list = []
    for i in np.arange(int(max(Y_))+1):
        kmf = KaplanMeierFitter()
        mask = y_test.loc[Y_==i, 'Outcome_NDI_death'].values
        ax = kmf.fit(y_test.loc[Y_==i, 'Outcome_NDI_Time'], y_test.loc[Y_==i, 'Outcome_NDI_death'], label=label_list[i]+" ("+str(round((sum(mask)*100)/len(mask), 2))+"% died)").plot_survival_function(ax=ax, color=color_list[i], ylabel="Estimated probability of survival", xlabel="Time in days from enrollment")
        kmf_list.append(kmf)
    add_at_risk_counts(*kmf_list, labels=label_list, ax=ax)
    ax.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.ylim((0.30, 1.00))
    chisq, pvalue = compare_survival(y=y_test.to_records(index=False), group_indicator=Y_)
    plt.title("Kaplan Meier curves (p = "+str(np.round(pvalue, 3))+")")
    
    plt.suptitle(set_name)
    plt.show()
    
def plot_results_HEART(X, Y_, title, y_test):
    color_list = np.array(["green", "orange", "red"])
    label_list = np.array(['Low risk', 'Moderate risk', 'High risk'])

    plt.figure(figsize=(8, 16))
    
    #PLOT 1
    plt.subplot(2, 1, 1)
    labels = [label_list[i]+' (n = '+str(sum(Y_==i))+')' for i in np.arange(3)]
    plt.hist([X[Y_==i] for i in np.arange(3)], 50, density=True, color=color_list, label=labels, stacked=True)
    plt.xlabel('HEART score')
    plt.ylabel('Density')
    plt.legend(loc="upper right")
    plt.title(title)
          
    a = np.isnan(Y_)
    if sum(a)!=0:
        Y_ = Y_[~a]
        y_test = y_test.iloc[~a]
    
    #PLOT 2
    
    ax = plt.subplot(2, 1, 2)
    kmf_list = []
    for i in np.arange(int(max(Y_))+1):
        kmf = KaplanMeierFitter()
        mask = y_test.loc[Y_==i, 'Outcome_NDI_death'].values
        ax = kmf.fit(y_test.loc[Y_==i, 'Outcome_NDI_Time'], y_test.loc[Y_==i, 'Outcome_NDI_death'], label=label_list[i]+" ("+str(round((sum(mask)*100)/len(mask), 2))+"% died)").plot_survival_function(ax=ax, color=color_list[i], ylabel="Estimated probability of survival", xlabel="Time in days from enrollment")
        kmf_list.append(kmf)
    add_at_risk_counts(*kmf_list, labels=label_list, ax=ax)
    ax.grid(True)
    plt.tight_layout()
    plt.ylim((0.30, 1.00))
    chisq, pvalue = compare_survival(y=y_test.to_records(index=False), group_indicator=Y_)
    plt.title("Kaplan Meier curves (p = "+str(np.round(pvalue, 3))+")")
    plt.show()
    
def plot_results_MEDIC(X, Y_, title , dpgmm):
    color_list = rearrange_colors(dpgmm.means_.squeeze())
    label_list = rearrange_colors(dpgmm.means_.squeeze(), np.array(['Low risk', 'Moderate risk', 'High risk']))
    
    plt.figure(figsize=(8, 8))
    
    x = np.linspace(min(X), max(X), 1000)
    pdf = np.exp(dpgmm.score_samples(x.reshape(-1, 1)))
    pdf_clusters = dpgmm.predict_proba(x.reshape(-1, 1)) * pdf[:, np.newaxis]
    
    plt.subplot(1, 1, 1)
    labels = [label_list[i]+' (n = '+str(sum(Y_==i))+')' for i in np.arange(3)]
    plt.hist([X[Y_==i] for i in np.arange(3)], 50, density=True, color=color_list, label=labels, stacked=True)
    plt.plot(x, pdf, '-k', linewidth=2)
    plt.plot(x, pdf_clusters, '--k', linewidth=2)
    plt.xlabel('Predicted score')
    plt.ylabel('Density')
    plt.legend(loc="best")
    plt.title(title)
    plt.show()
    
def RMSE_KM(y_test, survs, times):
    ts, prob_survival = kaplan_meier_estimator(y_test['Outcome_NDI_death'], y_test['Outcome_NDI_Time'])
    idx = (ts>=min(times)) & (ts<=max(times))
    ts = ts[idx]
    prob_survival = prob_survival[idx]
    preds_survs = np.row_stack([fn(ts) for fn in survs])
    rmse_survs = mean_squared_error(prob_survival, np.mean(preds_survs, axis=0), squared=False)
    return rmse_survs

# POST-HOC ANALYSIS

def plot_Sankey_diagram(clusters_HEART, clusters, y_test, dpgmm):
    pio.renderers.default = "browser"
    color_list = rearrange_colors(dpgmm.means_.squeeze()).tolist()
    data = pd.DataFrame.from_dict({
                                    'HEART': clusters_HEART,
                                    'PRED': clusters,
                                    'DEATH': y_test['Outcome_NDI_death'].values,
                                    })

    fig = go.Figure(data=[go.Sankey(
        node = dict(
            line = dict(color = "black", width = 0.5),
            x = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999],
            y = [0.001, 1/5, 2.5/5, 3.57/5, 4.75/5, 0.999, 0.001, 1/5, 2.25/5, 3/5, 4.25/5, 0.999],
            color = ['green', 'green', 'orange', 'orange', 'red', 'red',
                      'green', 'green', 'orange', 'orange', 'red', 'red'],
            ),
        link = dict(
            source = [0, 0, 0, 
                      1, 1, 1, 
                      2, 2, 2, 
                      3, 3, 3, 
                      4, 4, 4, 
                      5, 5, 5,
                      ],
            target = [6, 8, 10,
                      7, 9, 11,
                      6, 8, 10,
                      7, 9, 11,
                      6, 8, 10,
                      7, 9, 11,
                      ],
            value = [data.loc[(data['HEART']==0) & (data['PRED']==color_list.index('green')) & (data['DEATH']==True)].shape[0],
                     data.loc[(data['HEART']==0) & (data['PRED']==color_list.index('orange')) & (data['DEATH']==True)].shape[0],
                     data.loc[(data['HEART']==0) & (data['PRED']==color_list.index('red')) & (data['DEATH']==True)].shape[0],
                     
                     data.loc[(data['HEART']==0) & (data['PRED']==color_list.index('green')) & (data['DEATH']==False)].shape[0],
                     data.loc[(data['HEART']==0) & (data['PRED']==color_list.index('orange')) & (data['DEATH']==False)].shape[0],
                     data.loc[(data['HEART']==0) & (data['PRED']==color_list.index('red')) & (data['DEATH']==False)].shape[0],
                     
                     data.loc[(data['HEART']==1) & (data['PRED']==color_list.index('green')) & (data['DEATH']==True)].shape[0],
                     data.loc[(data['HEART']==1) & (data['PRED']==color_list.index('orange')) & (data['DEATH']==True)].shape[0],
                     data.loc[(data['HEART']==1) & (data['PRED']==color_list.index('red')) & (data['DEATH']==True)].shape[0],
                     
                     data.loc[(data['HEART']==1) & (data['PRED']==color_list.index('green')) & (data['DEATH']==False)].shape[0],
                     data.loc[(data['HEART']==1) & (data['PRED']==color_list.index('orange')) & (data['DEATH']==False)].shape[0],
                     data.loc[(data['HEART']==1) & (data['PRED']==color_list.index('red')) & (data['DEATH']==False)].shape[0],
                     
                     data.loc[(data['HEART']==2) & (data['PRED']==color_list.index('green')) & (data['DEATH']==True)].shape[0],
                     data.loc[(data['HEART']==2) & (data['PRED']==color_list.index('orange')) & (data['DEATH']==True)].shape[0],
                     data.loc[(data['HEART']==2) & (data['PRED']==color_list.index('red')) & (data['DEATH']==True)].shape[0],
                     
                     data.loc[(data['HEART']==2) & (data['PRED']==color_list.index('green')) & (data['DEATH']==False)].shape[0],
                     data.loc[(data['HEART']==2) & (data['PRED']==color_list.index('orange')) & (data['DEATH']==False)].shape[0],
                     data.loc[(data['HEART']==2) & (data['PRED']==color_list.index('red')) & (data['DEATH']==False)].shape[0],
                     ],
            color = ['rgba(255, 99, 71, 0.7)', 'rgba(255, 99, 71, 0.7)', 'rgba(255, 99, 71, 0.7)', 
                     'rgba(60, 179, 113, 0.3)', 'rgba(60, 179, 113, 0.3)', 'rgba(60, 179, 113, 0.3)',
                     'rgba(255, 99, 71, 0.7)', 'rgba(255, 99, 71, 0.7)', 'rgba(255, 99, 71, 0.7)', 
                     'rgba(60, 179, 113, 0.3)', 'rgba(60, 179, 113, 0.3)', 'rgba(60, 179, 113, 0.3)',
                     'rgba(255, 99, 71, 0.7)', 'rgba(255, 99, 71, 0.7)', 'rgba(255, 99, 71, 0.7)', 
                     'rgba(60, 179, 113, 0.3)', 'rgba(60, 179, 113, 0.3)', 'rgba(60, 179, 113, 0.3)',]
            )
        )])

    fig.update_layout(font_size=20, font_color="black", autosize=False, width=1200, height=700,)
    fig.show()

def plot_stacked_bar_graph(clusters, clusters_HEART, means):
    df_graph = pd.DataFrame(data=[[np.round(100*sum(clusters[clusters_HEART==i]==j)/sum(clusters_HEART==i), 2) for j in range(max(clusters)+1)] for i in range(3)],
                            columns=rearrange_colors(means, np.array(['Low Risk', 'Moderate Risk', 'High Risk']))).loc[:, np.array(['Low Risk', 'Moderate Risk', 'High Risk'])]
    df_graph.insert(0, 'HEART Score Risk Group', ['Low Risk', 'Moderate Risk', 'High Risk'])
    df_graph.plot(
                    x = 'HEART Score Risk Group', 
                    ylabel = 'Percentage (%)',
                    kind = 'barh', 
                    stacked = True, 
                    title = 'Reclassificaton performance (Ref: HEART Score)', 
                    mark_right = True,
                    color = np.array(["green", "orange", "red"])
                    )
    for n in df_graph.columns[1:]:
        for i, (cs, ab) in enumerate(zip(df_graph.iloc[:, 1:].cumsum(1)[n], df_graph[n])):
            plt.text(cs - ab / 2, i, str(ab) + ' %', 
                      va = 'center', ha = 'center', color='black', weight='bold')
    plt.show()

def plot_Shapley_explainability(clf, df_test, cols, set_name, nf, masker_data):
    shap.initjs()
    df = pd.DataFrame(data=df_test, columns=(cols))
    masker = shap.maskers.Independent(data=masker_data)
    explainer = shap.Explainer(clf.predict, masker = masker) #clf: trained classifier
    shap_values = explainer(df)
    shap_duration = time.time() - start_time
    print("--- %s minutes %s seconds ---" % (shap_duration // 60, shap_duration % 60))
    # Shapley explainability - Plot 1
    plt.figure()
    shap.summary_plot(shap_values, features=df, feature_names=np.array(
        cols), sort=True, show=False, max_display=nf, color_bar=True, plot_type='dot')
    plt.title('Shapley values of the top '+str(nf)+'/'+str(len(cols)) +
              ' features for mortality prediction ('+set_name+')', fontweight="bold")
    plt.show()
    # Shapley explainability - Plot 2
    plt.figure()
    shap.summary_plot(shap_values, features=df, feature_names=np.array(
        cols), sort=True, show=False, max_display=nf, color_bar=True, plot_type='bar')    
    plt.title('Shapley values of the top '+str(nf)+'/'+str(len(cols)) +
              ' features for mortality prediction ('+set_name+')', fontweight="bold")
    plt.show()
    
if __name__ == "__main__":
    
    #Variables
    inputs=parse_args()
    num_folds=inputs['num_folds']
    imputation_method=inputs['imputation_method']
    clf_name=inputs['clf_name']
    
    dfy = get_y()
    dfX = get_X()
    n1 = dfy.shape[0]
    dfX, dfy = remove_missing_patients(dfX, dfy)
    n_miss, p_miss = n1 - dfy.shape[0], (n1 - dfy.shape[0])/n1
    dfX, dfy = shuffle(dfX, dfy, random_state=0)
    
    # Set evaluation times
    lower, upper = np.percentile(dfy['Outcome_NDI_Time'], [10, 90])
    times = np.arange(lower, upper + 1)
   
    X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size=0.2, stratify = dfy['Outcome_NDI_death'], random_state = 99)
    
    imputer = {
                'mean': SimpleImputer(strategy='mean', add_indicator=True),
                'knn': KNNImputer(n_neighbors=5,
                                  weights='uniform',
                                  add_indicator=True),
                'iterative': IterativeImputer(verbose=0,
                                              random_state=0,
                                              add_indicator=True),
                }
    classifier = {
                #Ensemble
                'Componentwise_GBMS': ComponentwiseGradientBoostingSurvivalAnalysis(loss='coxph', random_state=0),
                'GBMS': GradientBoostingSurvivalAnalysis(loss='coxph', random_state=0),
                'RSF': RandomSurvivalForest(n_jobs=-1, random_state=0),
                'EST': ExtraSurvivalTrees(n_jobs=-1, random_state=0),
                #Linear
                'Coxnet': CoxnetSurvivalAnalysis(fit_baseline_model=True),
                'CoxPH': CoxPHSurvivalAnalysis(alpha=0.0001),
                }
    estimator = Pipeline([
                        ('normalizer', StandardScaler()),
                        ('imputer', imputer[imputation_method]),
                        ('clf', as_concordance_index_ipcw_scorer(classifier[clf_name], tau=times[-1]))
                        ])
     
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  
    
    rsf = estimator.fit(X_train, y_train.to_records(index=False))
    
    cindex = concordance_index_ipcw(y_train.to_records(index=False), y_test.to_records(index=False), rsf.predict(X_test), tau=times[-1])
    cdauc = cumulative_dynamic_auc(y_train.to_records(index=False), y_test.to_records(index=False), rsf.predict(X_test), times)
    
    survs = rsf.predict_survival_function(X_test)
    preds_survs = np.row_stack([fn(times) for fn in survs])
    rmse_survs = RMSE_KM(y_test, survs, times)
    
    ibs = integrated_brier_score(y_train.to_records(index=False), y_test.to_records(index=False), preds_survs, times)

    ### CLUSTERING ###
    
    scores = cross_val_predict(estimator, X_train, y_train.to_records(index=False), cv=skf.split(X_train, y_train['Outcome_NDI_death']))
    X = scores
    dpgmm = BayesianGaussianMixture(n_components=3, covariance_type="full", random_state=0).fit(X.reshape(-1, 1))
    clusters_CV = dpgmm.predict(X.reshape(-1, 1))
    plot_results(
                X,
                clusters_CV,
                "Bayesian GMM-based Clustering",
                y_train,
                dpgmm,
                "Cross-Validation Scores"
                )
    ss = {}
    ss["CV"] = silhouette_score(X.reshape(-1, 1), clusters_CV, random_state=0)
    
    scores = rsf.predict(X_test)
    X = scores
    clusters = dpgmm.predict(X.reshape(-1, 1))
    (pd.DataFrame(data=clusters, index=X_test.index, columns=["Cluster"])).to_csv("Cluster_Results_Internal_Testing.csv")
    plot_results(
                X,
                clusters,
                "Bayesian GMM-based Clustering",
                y_test,
                dpgmm,
                "Testing Scores"
                )
    ss["Test"] = silhouette_score(X.reshape(-1, 1), clusters, random_state=0)
    clusters_HEART, HEART_score = get_clusters_HEART(y_test)
    plot_results_HEART(
                    HEART_score,
                    clusters_HEART,
                    "HEART Score-based Clustering",
                    y_test,
                    )
    
    plot_stacked_bar_graph(clusters, clusters_HEART, dpgmm.means_.squeeze())
    
    X_MEDIC = get_X_MEDIC()
    X_MEDIC = remove_missing_patients(X_MEDIC)
    X_MEDIC = clean_outliers(X_MEDIC)
    X_MEDIC = shuffle(X_MEDIC, random_state=0)
    scores = rsf.predict(X_MEDIC)
    X = scores
    clusters_MEDIC = dpgmm.predict(X.reshape(-1, 1))
    (pd.DataFrame(data=clusters_MEDIC, index=X_MEDIC.index, columns=["Cluster"])).to_csv("Cluster_Results_External_Validation.csv")
    plot_results_MEDIC(
                        X,
                        clusters_MEDIC,
                        "MEDIC dataset: Bayesian GMM-based Clustering",
                        dpgmm,
                        )
    ss["MEDIC"] = silhouette_score(X.reshape(-1, 1), clusters_MEDIC, random_state=0)
    
    prog_duration = time.time() - start_time
    print("--- %s minutes %s seconds ---" % (prog_duration // 60, prog_duration % 60))