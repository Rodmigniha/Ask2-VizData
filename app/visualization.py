import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis
from scipy.cluster.hierarchy import dendrogram, linkage
import streamlit as st
from sklearn.linear_model import LinearRegression


def plot_histogram(data, column):
    fig=plt.figure(figsize=(8, 5))
    sns.histplot(data[column], kde=False, bins=30, color='blue')
    plt.title(f'Histogramme de {column}')
    st.pyplot(fig)

def plot_boxplot(data, column):
    fig=plt.figure(figsize=(6, 4))
    sns.boxplot(y=data[column], color='blue')
    plt.title(f'Boxplot de {column}')
    st.pyplot(fig)


def plot_kde(data, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data[column], fill=True, color='blue')
    plt.title(f'Distribution KDE de {column}')
    st.pyplot(fig)


def plot_pareto(data, column):
    counts = data[column].value_counts()
    cumulative = counts.cumsum() / counts.sum()
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(counts.index, counts, color='blue')
    ax1.set_ylabel('Fréquence')
    ax2 = ax1.twinx()
    ax2.plot(counts.index, cumulative, color='red', marker='o', linestyle='dashed')
    ax2.set_ylabel('Fréquence cumulative')
    plt.title(f'Pareto Chart de {column}')
    st.pyplot(fig)


def plot_scatter(data, x, y):
    fig=plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data[x], y=data[y], color='blue')
    plt.title(f'Scatter plot entre {x} et {y}')
    st.pyplot(fig)


def plot_violin(data, x, y):
    fig=plt.figure(figsize=(8, 5))
    sns.violinplot(x=data[x], y=data[y], hue=data[x], palette='Blues', legend=False)
    plt.title(f'Violin plot de {y} selon {x}')
    st.pyplot(fig)


def plot_correlation_heatmap(data):
    fig=plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='Blues', fmt='.2f')
    plt.title('Heatmap des corrélations')
    st.pyplot(fig)


def plot_hexbin(data, x, y):
    fig=plt.figure(figsize=(8, 5))
    plt.hexbin(data[x], data[y], gridsize=30, cmap='Blues')
    plt.colorbar(label='Densité')
    plt.title(f'Hexbin plot de {x} et {y}')
    st.pyplot(fig)


def plot_line_chart(data, x, y):
    fig=plt.figure(figsize=(8, 5))
    sns.lineplot(x=data[x], y=data[y], color='blue')
    plt.title(f'Évolution de {y} au cours de {x}')
    st.pyplot(fig)


def plot_lorenz_curve(data, column):
    sorted_data = np.sort(data[column].dropna())
    cum_dist = np.cumsum(sorted_data) / np.sum(sorted_data)
    
    fig=plt.figure(figsize=(8, 5))
    plt.plot(np.linspace(0, 1, len(cum_dist)), cum_dist, color='blue', label='Lorenz Curve')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Égalité parfaite')
    plt.title(f'Courbe de Lorenz pour {column}')
    plt.legend()
    st.pyplot(fig)


def plot_dendrogram(data, columns):
    Z = linkage(data[columns].dropna(), method='ward')
    fig=plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title('Dendrogramme des clusters')
    st.pyplot(fig)


def plot_missing_values_bar(data):
    missing = data.isnull().sum()
    missing = missing[missing > 0]  # Ne garder que les colonnes avec des valeurs manquantes
    
    if missing.empty:
        st.write("Aucune valeur manquante détectée.")
        return

    fig=plt.figure(figsize=(8, 5))
    missing.plot(kind='bar', color='blue')
    plt.xlabel("Colonnes")
    plt.ylabel("Nombre de valeurs manquantes")
    plt.title("Nombre de valeurs manquantes par colonne")
    st.pyplot(fig)


def plot_missing_values_heatmap(data):
    if data.isnull().sum().sum() == 0:
        st.write("Aucune valeur manquante détectée.")
        return

    fig=plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cmap='Blues', cbar=False)
    plt.title("Carte des valeurs manquantes")
    st.pyplot(fig)


def moving_average(data, window=5):
    return data.rolling(window=window).mean()


def linear_trend(data):
    X = np.arange(len(data)).reshape(-1, 1)  
    y = data.values.reshape(-1, 1)  
    model = LinearRegression().fit(X, y)
    trend = model.predict(X).flatten()  
    return trend


def plot_stacked_histogram(df, column, group_col, bins=20):
    fig=plt.figure(figsize=(8, 5))
    unique_groups = df[group_col].unique()
    
    for group in unique_groups:
        subset = df[df[group_col] == group][column]
        plt.hist(subset, bins=bins, alpha=0.6, label=f"Groupe {group}", edgecolor="black")
    
    plt.title("Histogramme empilé")
    plt.legend()
    st.pyplot(fig)


def plot_line_chart_with_trend(data, window=5):
    fig=plt.figure(figsize=(10, 5))
    
    
    plt.plot(data.index, data, label="Données originales", color="blue", alpha=0.6)

    
    ma = moving_average(data, window=window)
    plt.plot(data.index, ma, label=f"Moyenne Mobile ({window} jours)", color="orange", linewidth=2)

    
    trend = linear_trend(data)
    plt.plot(data.index, trend, label="Tendance Linéaire", color="red", linestyle="dashed", linewidth=2)

    plt.xlabel("Temps")
    plt.ylabel("Valeurs")
    plt.title("Line Chart avec Moyenne Mobile et Tendance Linéaire")
    plt.legend()
    plt.grid()
    st.pyplot(fig)

def plot_histogram_with_density(data, bins=20):
    fig=plt.figure(figsize=(8, 5))
    sns.histplot(data, bins=bins, kde=True, color="blue", alpha=0.6)
    plt.title("Histogramme avec densité")
    plt.xlabel("Valeurs")
    plt.ylabel("Fréquence")
    plt.grid()
    st.pyplot(fig)
    

def plot_pie_chart(data, column):
    fig=plt.figure(figsize=(6, 6))
    data[column].value_counts().plot.pie(autopct='%1.1f%%', cmap='Blues', startangle=90)
    plt.ylabel("")  # Supprime le label de l'axe Y
    plt.title(f"Répartition de {column}")
    st.pyplot(fig)
    
    
def plot_bar_chart_top10(data, column):
    top10 = data[column].value_counts().nlargest(10)  # Prend les 10 catégories les plus fréquentes
    fig = plt.figure(figsize=(8, 5))
    sns.barplot(x=top10.values, y=top10.index, palette='Blues_r')
    plt.xlabel("Nombre d'occurrences")
    plt.ylabel(column)
    plt.title(f"Top 10 des catégories de {column}")
    st.pyplot(fig)

def visualiz(filtered_data, numerical_colonnes_valides, categorical_colonnes_valides, chart_type_concernes):
    
    n=len(numerical_colonnes_valides)
    c=len(categorical_colonnes_valides)
    if n>=1:
        for i in range(n):
            col = numerical_colonnes_valides[i]
            for chart in chart_type_concernes:
                if chart =="plot_histogram":
                    plot_histogram(filtered_data, col)
                elif chart =="plot_boxplot":
                    plot_boxplot(filtered_data, col)
                elif chart =="plot_kde":
                    plot_kde(filtered_data, col)
                elif chart =="plot_pareto":
                    plot_pareto(filtered_data, col)
                elif chart =="plot_scatter":
                    if i<n-1:
                        for j in range(i+1,n):
                            plot_scatter(filtered_data, col, numerical_colonnes_valides[j])
                elif chart =="plot_violin":
                    if i<n-1:
                        for j in range(i+1,n):
                            plot_violin(filtered_data, col, numerical_colonnes_valides[j])
                elif chart =="plot_correlation_heatmap":
                    plot_correlation_heatmap(filtered_data[numerical_colonnes_valides])
                elif chart =="plot_hexbin":
                    if c > 0 :
                        for j in range(c):
                            plot_hexbin(filtered_data, col, categorical_colonnes_valides[j])
                elif chart =="plot_line_chart":
                    if i<n-1:
                        for j in range(i+1,n):
                            plot_line_chart(filtered_data, col, numerical_colonnes_valides[j])
                elif chart =="plot_lorenz_curve":
                    plot_lorenz_curve(filtered_data, col)
                elif chart == "plot_line_chart_with_trend":
                    plot_line_chart_with_trend(col, window=5)
                elif chart == "plot_histogram_with_density":
                    plot_histogram_with_density(col, bins=20)
    
    if c >0 :
        for i in range(c):
            col = categorical_colonnes_valides[i]
            for chart in chart_type_concernes: 
                if chart == "plot_pie_chart":
                    plot_pie_chart(filtered_data,col)
                elif chart == "plot_bar_chart_top10":
                    plot_bar_chart_top10(filtered_data,col)
                elif chart =="plot_violin":
                    if n>0:
                        for j in range(n):
                            plot_violin(filtered_data, col, numerical_colonnes_valides[j])
                    
        
                
                
                

