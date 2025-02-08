def mapp_viz():
    
    mapping = {
        "1- Distribution d'une variable": {
            "Indicateurs": ["Moyenne", "Médiane", "Écart-type", "Quartiles", "Histogramme des fréquences"],
            "Graphiques": ["Histogramme", "Boxplot"],
            "Calculs_fct": ["mean", "median", "variance", "std_dev", "iqr", "min_max", "quantiles"],
            "Graphiques_fct": ["plot_histogram", "plot_boxplot", "plot_kde", "plot_ecdf", "plot_pareto"]
        },
        "2- Existence de valeurs aberrantes dans une variable": {
            "Indicateurs": ["Plages interquartiles (IQR)", "Z-score", "Tukey's Fences"],
            "Graphiques": ["Boxplot", "Scatter plot", "Distribution KDE"],
            "Calculs_fct": ["z_score", "mahalanobis_distance", "isolation_forest", "lof"],
            "Graphiques_fct": ["plot_scatter", "plot_boxplot"]
        },
        "3- Tendance centrale de la variable X selon la catégorie Y": {
            "Indicateurs": ["Moyenne", "Médiane", "Mode par groupe"],
            "Graphiques": ["Bar chart", "Violin plot", "Boxplot"],
            "Calculs_fct": ["t_test", "mann_whitney_u", "anova", "kruskal_wallis"],
            "Graphiques_fct": ["plot_violin", "plot_swarm", "plot_bar_chart", "plot_stacked_bar_chart", "plot_mosaic", "plot_histogram"]
            
        },
        "4- Corrélation entre X et Y": {
            "Indicateurs": ["Coefficient de corrélation de Pearson", "Coefficient de corrélation de Spearman"],
            "Graphiques": ["Scatter plot avec ligne de tendance", "Heatmap des corrélations"],
            "Calculs_fct": ["pearson_corr", "spearman_corr"],
            "Graphiques_fct": ["plot_scatter_trend", "plot_correlation_heatmap", "plot_hexbin", "plot_2d_density", "plot_boxplot"]
        },
        "5- Evolution d'une variable X au cours du temps": {
            "Indicateurs": ["Moyenne", "Médiane", "Min/Max sur différentes périodes"],
            "Graphiques": ["Line chart", "Area chart"],
            "Calculs_fct": ["moving_average", "linear_trend", "decompose_series", "adf_test"],
            "Graphiques_fct": ["plot_line_chart", "plot_area_chart", "plot_line_chart_regression", "plot_line_chart_decomposition", "plot_autocorrelogram"]
        },
        "6- Répartition des valeurs d'une variable catégorielle Y": {
            "Indicateurs": ["Fréquences", "Proportions"],
            "Graphiques": ["Bar chart", "Pie chart"],
            "Calculs_fct": ["frequency_table", "gini_coefficient", "theil_index"],
            "Graphiques_fct": ["plot_pie_chart", "plot_bar_chart_top10"]
        },
        "7- Probabilité qu'un individu ait une valeur X supérieure à un certain seuil": {
            "Indicateurs": ["Loi de probabilité", "Quantiles", "Densité"],
            "Graphiques": ["Distribution KDE", "Histogramme avec densité"],
            "Calculs_fct": ["estimate_distribution", "compute_quantiles"],
            "Graphiques_fct": ["plot_kde", "plot_histogram_with_density"]
            
        },
        "8- Quelle est la variabilité de X selon Y ?": {
            "Indicateurs": ["Variance", "Écart-type par groupe"],
            "Graphiques": ["Boxplot", "Violin plot", "Swarm plot"],
            "Calculs_fct": ["variance", "std_dev"],
            "Graphiques_fct": ["plot_boxplot", "plot_violin"]
        },
        "9- Quelle est la tendance de un indicateur sur une période donnée ?": {
            "Indicateurs": ["Moyenne mobile", "Tendance linéaire"],
            "Graphiques": ["Line chart avec régression linéaire"],
            "Calculs_fct": ["moving_average", "linear_trend"],
            "Graphiques_fct": ["plot_line_chart_with_trend"]
        },
        "10- Part cumulative de X ": {
            "Indicateurs": ["Fonction de répartition empirique (ECDF)", "Percentiles"],
            "Graphiques": ["Courbe ECDF", "Pareto Chart"],
            "Calculs_fct": [""],
            "Graphiques_fct": ["plot_ecdf", "plot_pareto"]
        },
        "11- Comment la distribution de X diffère-t-elle entre différentes sous-populations ?": {
            "Indicateurs": ["Comparaison de moyennes", "ANOVA", "Test de Mann-Whitney"],
            "Graphiques": ["Boxplot", "Violin plot", "Histogramme empilé"],
            "Calculs_fct": ["mann_whitney_test", "anova_test", "compare_means"],
            "Graphiques_fct": ["plot_stacked_histogram", "plot_boxplot", "plot_violin"]
        },
        "12- Existence de saisonnalité dans les données temporelles de X": {
            "Indicateurs": ["Décomposition de séries temporelles", "Test de stationnarité"],
            "Graphiques": ["Line chart avec décomposition", "Autocorrelogramme"],
            "Calculs_fct": ["decompose_series", "adf_test"],
            "Graphiques_fct": ["plot_decomposition", "plot_autocorrelogram"]
        },
        "13- Fréquence des valeurs manquantes dans chaque variable ": {
            "Indicateurs": ["Nombre de valeurs manquantes par colonne"],
            "Graphiques": ["Bar chart", "Heatmap des valeurs manquantes"],
            "Calculs_fct": ["nombre_valeurs_manquantes"],
            "Graphiques_fct": ["plot_missing_values_bar", "plot_missing_values_heatmap"]
        },
        "14- Comment varie la proportion de la catégorie Y en fonction de X ?": {
            "Indicateurs": ["Taux de variation", "Ratios"],
            "Graphiques": ["Stacked Bar Chart", "Mosaic plot"],
            "Calculs_fct": ["taux_de_variation", "calcul_ratio"],
            "Graphiques_fct": ["plot_stacked_bar", "plot_mosaic"]
            
        },
        "15- Impact de une variable catégorielle sur une variable quantitative ?": {
            "Indicateurs": ["Test de Kruskal-Wallis", "ANOVA"],
            "Graphiques": ["Boxplot par catégorie", "Violin plot"],
            "Calculs_fct": ["kruskal_wallis", "anova"],
            "Graphiques_fct": ["plot_boxplot", "plot_violin"]
        },
        "16- Quelle est la distribution jointe de X et Y ?": {
            "Indicateurs": ["Densité conjointe", "Loi jointe"],
            "Graphiques": ["Hexbin plot", "2D Density plot"],
            "Calculs_fct": [""],
            "Graphiques_fct": ["plot_hexbin"]
        },
        "17- Quel est le top 10 des valeurs les plus fréquentes pour la variable catégorielle Y ?": {
            "Indicateurs": ["Fréquences absolues et relatives"],
            "Graphiques": ["Bar chart", "Pareto chart"],
            "Calculs_fct": ["frequency_table"],
            "Graphiques_fct": ["plot_pareto","plot_bar_chart"]
        },
        "18- Y a-t-il une différence significative entre deux groupes pour la variable X ?": {
            "Indicateurs": ["Test t de Student", "Test U de Mann-Whitney"],
            "Graphiques": ["Boxplot", "Violin plot avec test statistique"],
            "Calculs_fct": ["t_test", "mann_whitney_u", "anova", "kruskal_wallis"],
            "Graphiques_fct": ["plot_violin", "plot_boxplot"]
        },
        "19- Peut-on regrouper les observations en clusters basés sur X et Y ?": {
            "Indicateurs": ["Analyse en clusters (K-means, DBSCAN)"],
            "Graphiques": ["Scatter plot avec clustering", "Dendrogramme"],
            "Calculs_fct": ["kmeans_clustering", "dbscan_clustering"],
            "Graphiques_fct": ["plot_scatter_clustering", "plot_dendrogram"]
        },
        "20- Quelle est la contribution cumulative des premières modalités de Y ?": {
            "Indicateurs": ["Cumulative Frequency"],
            "Graphiques": ["Courbe de Lorenz", "Pareto Chart"],
            "Calculs_fct": ["cumulative_frequency"],
            "Graphiques_fct": ["plot_pareto", "plot_lorenz_curve"]
        },
        "21- Peut-on détecter une anomalie dans les données ?": {
            "Indicateurs": ["Distance de Mahalanobis", "Isolation Forest", "LOF"],
            "Graphiques": ["Scatter plot avec anomalies", "Boxplot"],
            "Calculs_fct": ["z_score", "mahalanobis_distance", "isolation_forest", "lof"],
            "Graphiques_fct": ["plot_scatter", "plot_boxplot"]
        },
        "22- Quelle est la concentration des valeurs sur une variable numérique ?": {
            "Indicateurs": ["Coefficient de Gini", "Indice de Theil"],
            "Graphiques": ["Courbe de Lorenz", "Histogramme"],
            "Calculs_fct": ["cumulative_frequency"],
            "Graphiques_fct": ["plot_lorenz_curve"]
        },
         "23- Structure du dataset": {  
            "Calculs_fct": ["shape", "describe", "info"],
            "Graphiques_fct": []
        }
        
        
    }
    return mapping




