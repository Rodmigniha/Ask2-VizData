import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import streamlit as st


from scipy.stats import norm

from scipy.stats import ttest_ind, f_oneway, mannwhitneyu


import streamlit as st

def mean(series):
    return np.mean(series)


def median(series):
    return np.median(series)


def mode(series):
    return series.mode().values[0] if not series.mode().empty else None


def variance(series):
    return np.var(series, ddof=1)


def std_dev(series):
    return np.std(series, ddof=1)


def iqr(series):
    return stats.iqr(series)


def min_max(series):
    return series.min(), series.max()


def quantiles(series, q=[0.25, 0.5, 0.75]):
    return series.quantile(q).to_dict()


def percentiles(series, p=95):
    return np.percentile(series, p)


def probability_distribution(series):
    return stats.norm.fit(series)


def kde(series):
    sns.kdeplot(series)
    plt.show()


def histogram(series, bins=10):
    plt.hist(series, bins=bins)
    plt.show()



def pearson_corr(series1, series2):
    return stats.pearsonr(series1, series2)[0]


def spearman_corr(series1, series2):
    return stats.spearmanr(series1, series2)[0]


def t_test(series1, series2):
    return stats.ttest_ind(series1, series2)


def mann_whitney_u(series1, series2):
    return stats.mannwhitneyu(series1, series2)


def anova(*groups):
    return stats.f_oneway(*groups)


def kruskal_wallis(*groups):
    return stats.kruskal(*groups)


def moving_average(series, window=3):
    return series.rolling(window=window, min_periods=1).mean()


def l_trend(series):
    x = np.arange(len(series))
    slope, intercept, _, _, _ = stats.linregress(x, series)
    return slope, intercept


def linear_trend(series):
    slope, intercept = l_trend(series)  
    st.write(f"Tendance Linéaire de la colonne gmrt_in_air10 : y = {slope:.3f} * x + {intercept:.3f}")


def missing_values(df):
    return df.isnull().sum()


def mahalanobis_distance(df, ref=None):
    cov = EmpiricalCovariance().fit(df)
    return cov.mahalanobis(df - ref if ref is not None else df)


def isolation_forest(df, contamination=0.05):
    iso = IsolationForest(contamination=contamination)
    return iso.fit_predict(df)


def lof(df, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    return lof.fit_predict(df)


def z_score(series):
    return np.abs(stats.zscore(series))


def frequency_table(series):
    return series.value_counts(normalize=True)


def gini_coefficient(series):
    values = series.value_counts(normalize=True).values
    return 1 - np.sum(values**2)


def theil_index(series):
    mean = series.mean()
    return np.sum((series / mean) * np.log(series / mean)) / len(series)


def cumulative_frequency(series):
    return series.value_counts().sort_index().cumsum()


def kmeans_clustering(df, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    return model.fit_predict(df)


def dbscan_clustering(df, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(df)



def taux_de_variation(series):
    return series.pct_change() * 100  


def calcul_ratio(data, num_col, denom_col):
    return data[num_col] / data[denom_col]


def nombre_valeurs_manquantes(data):
    return data.isnull().sum()


def compare_means(group1, group2):
    stat, p_value = ttest_ind(group1, group2, equal_var=False)
    return {"T-statistic": stat, "p-value": p_value}



def anova_test(*groups):
    stat, p_value = f_oneway(*groups)
    return {"F-statistic": stat, "p-value": p_value}


def mann_whitney_test(group1, group2):
    stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
    return {"U-statistic": stat, "p-value": p_value}


def estimate_distribution(data):
    mu, sigma = norm.fit(data) 
    return mu, sigma


def compute_quantiles(data, quantiles=[0.25, 0.5, 0.75]):
    return np.quantile(data, quantiles)


def statistic_indic(filtered_data, numerical_colonnes_valides, Mesure_mapp_concernes):
    m=len(numerical_colonnes_valides)
    if m>=1:
        for i in range(m):
            col = filtered_data[numerical_colonnes_valides[i]]
            col_i = numerical_colonnes_valides[i]
            for mes in Mesure_mapp_concernes:
                if mes =="mean":
                    st.write(f'Moyenne de la colonne {col_i} : {mean(col)}')
                elif mes =="median":
                    st.write(f'Médiane de la colonne {col_i} : {median(col)}')
                elif mes =="mode":
                    st.write(f'Mode de la colonne {col_i} :  {mode(col)}')
                elif mes =="variance":
                    st.write(f'variance de la colonne {col_i} :  { variance(col)}')
                elif mes =="std_dev":
                    st.write(f'Ecart-type de la colonne {col_i} :  {std_dev(col)}')
                elif mes =="iqr":
                    st.write(f'IQR de la colonne {col_i} :  {iqr(col)}')
                elif mes =="min_max":
                    st.write(f'Min-Max de la colonne {col_i} :  {min_max(col)}')
                elif mes =="quantile":
                    st.write(f'Quantiles de la colonne {col_i} :  {quantiles(col, q=[0.25, 0.5, 0.75])}')
                elif mes =="percentiles":    
                    st.write(f'Percentiles de la colonne {col_i} :  { percentiles(col, p=95)}')
                elif mes =="probability_distribution":    
                    st.write(f'Distribution de probabilité de la colonne {col_i} : {  probability_distribution(col)}')
                elif mes =="kde":    
                    st.write(f'KDE de la colonne {col_i} :  {  kde(col)}')
                elif mes =="histogram":
                    st.write(f'Histogramme de la colonne {col_i} :  {  histogram(col, bins=10)}')
                elif mes =="ecdf":    
                    st.write(f'Histogramme de la colonne {col_i} :  {  ecdf(col)}')
                elif mes =="pearson_corr":
                    if i<m-1:
                        for j in range(i+1,m):
                            st.write(f'Correlation de Pearson de {col_i} et {numerical_colonnes_valides[j]} :  { pearson_corr(col, filtered_data[numerical_colonnes_valides[j]])}')
                elif mes =="spearman_corr":
                    if i<m-1:
                        for j in range(i+1,m):
                            st.write(f'Correlation de Spearman de {col_i} et {numerical_colonnes_valides[j]} :  {  spearman_corr(col, filtered_data[numerical_colonnes_valides[j]])}')
                elif mes =="t_test":
                    if i<m-1:
                        for j in range(i+1,m):
                            st.write(f't_test de {col_i} et {numerical_colonnes_valides[j]} :  {  t_test(col, filtered_data[numerical_colonnes_valides[j]])}')
                elif mes =="mann_whitney_u":
                    if i<m-1:
                        for j in range(i+1,m):
                            st.write(f'mann_whitney_u de {col_i} et {numerical_colonnes_valides[j]} :  { mann_whitney_u(col, filtered_data[numerical_colonnes_valides[j]])}')
                elif mes =="moving_average":
                    st.write(f'Moving Average de la colonne {col_i} :  { moving_average(col, window=3)}')
                elif mes =="linear_trend":  
                    st.write(f'Tendance Linéaire de la colonne {col_i} :  {linear_trend(col)}')
                
                elif mes =="missing_values":
                    st.write(f'Valeurs manquantes :  {missing_values(filtered_data)}')
                elif mes =="mahalanobis_distance":
                    st.write(f'Distance de Mahalanobis :  {mahalanobis_distance(filtered_data, ref=None)}')
                elif mes =="isolation_forest":
                    st.write(f'isolation_forest :  { isolation_forest(filtered_data, contamination=0.05)}')
                elif mes =="lof":
                    st.write(f'lof :  { lof(filtered_data, n_neighbors=20)}')
                elif mes =="z_score":
                    st.write(f'z_score de {col_i} :  {  z_score(col)}')
                elif mes == "frequency_table":
                    st.write(f'Table de Frequence ce {col_i}:  { frequency_table(col)}')
                elif mes=="gini_coefficient":
                    st.write(f'Gini coefficient de {col_i} :  { gini_coefficient(col)}')
                elif mes =="theil_index":
                    st.write(f'Theil Index de {col_i}:  { theil_index(col)}')
                elif mes =="cumulative_frequency":
                    st.write(f'Cumumative Frequency de {col_i}:  { cumulative_frequency(col)}')
                elif mes =="kmeans_clustering":
                    st.write(f'kmeans_clustering:  { kmeans_clustering(filtered_data, n_clusters=3)}')
                elif mes =="taux_de_variation":
                    st.write(f'Taux de variation de {col_i}:  { taux_de_variation(col)}')    
                elif mes =="nombre_valeurs_manquantes":
                    st.write(f'Nombre valeurs manquantes de {col_i}:  { nombre_valeurs_manquantes(col)}')
                elif mes =="estimate_distribution":
                    st.write(f'Estimate distribution de {col_i}:  { estimate_distribution(col)}')
                elif mes =="compute_quantiles":
                    st.write(f'Quantiles de {col_i}:  { compute_quantiles(col, quantiles=[0.25, 0.5, 0.75])}')
                
                    
    
 