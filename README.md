# DataViz Genius

## Description

DataViz Genius est une application interactive de visualisation de données permettant d'explorer n'importe quel dataset tabulaire. Grâce à une interface intuitive, elle permet de poser des questions sur les données et d'obtenir des visualisations pertinentes et des statistiques détaillées en réponse.

L'application utilise **Streamlit** pour l'interface, **gemini** pour l'interprétation des questions, ainsi que **Seaborn**, **Matplotlib** et **Scikit-learn** pour les analyses et les graphiques.

## Fonctionnalités

- **Importation de fichiers CSV ou Excel**
- **Nettoyage et normalisation des données**
- **Filtrage dynamique des données**
- **Interprétation automatique des questions utilisateur**
- **Génération de visualisations adaptées à la question posée**
- **Affichage des statistiques descriptives clés**
- **Utilisation d'un modèle d'IA pour mapper les questions aux graphiques appropriés**

## Technologies utilisées

- **Python** (pandas, numpy, seaborn, matplotlib, scikit-learn)
- **Streamlit** (interface utilisateur)
- **GOOGLE API**  (interprétation des questions)
- **Dotenv** (gestion des variables d'environnement)
- **GitHub** (gestion de version et collaboration)

## Installation

### Prérequis

- Python 3.8+
- Poetry (gestionnaire de dépendances recommandé)

### Étapes d'installation

1. **Cloner le projet** :

   ```bash
   git clone https://github.com/Rodmigniha/Ask2-VizData.git
   cd dataviz-genius
   ```

2. **Créer un environnement virtuel et installer les dépendances** :

   ```bash
   poetry install
   ```

3. **Configurer la clé API** :

   - Créez un fichier `.env` à la racine du projet et ajoutez votre clé API :
     ```bash
     GOOGLE_API_KEY=VOTRE_CLE_API
     ```

4. **Lancer l'application** :

   ```bash
   streamlit run app/main.py
   ```

## Utilisation

1. Téléversez un fichier CSV ou Excel dans la barre latérale
2. Posez une question sur les données (ex : *Quelle est la moyenne de la colonne X ?*)
3. Observez les statistiques et visualisations générées dynamiquement

## Exemples de questions

- "Quelle est la répartition de la variable par catégorie ?"
- "Quel est la moyenne de la variable X ?"

## Quelques illustrations

![Dashboard principal](https://github.com/Rodmigniha/Ask2-VizData/blob/main/app/data/dsh1.PNG)
![Dashboard principal](https://github.com/Rodmigniha/Ask2-VizData/blob/main/app/data/dsh2.PNG)
![Dashboard principal](https://github.com/Rodmigniha/Ask2-VizData/blob/main/app/data/dsh3.PNG)
![Dashboard principal](https://github.com/Rodmigniha/Ask2-VizData/blob/main/app/data/dsh4.PNG)
![Dashboard principal](https://github.com/Rodmigniha/Ask2-VizData/blob/main/app/data/dsh5.PNG)

## Contribution

Les contributions sont les bienvenues ! Veuillez suivre ces étapes :

1. Forkez le projet
2. Créez une branche (`feature-nouvelle-fonctionnalite`)
3. Faites vos modifications et committez (`git commit -m "Ajout d'une nouvelle fonctionnalité"`)
4. Poussez la branche et ouvrez une pull request

## Auteurs

- **Rodrigue MIGNIHA** 
- **Lina THABET** 



