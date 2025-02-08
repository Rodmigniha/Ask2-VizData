import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os
import requests
import json
import google.generativeai as genai
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.llms import GoogleGenerativeAI


from visualization import visualiz

from ai_interface import (
    extract_columns_from_response,
    extract_question_from_response,
    )

from mapping import mapp_viz
from statistic_indicator import statistic_indic

from dataload import load_data

st.title("Ask 2 DataVIZ")
st.write(
        """
        Cette application permet d'analyser et de visualiser n'importe quel dataset tabulaire
        et bien plus encore ! 
        """
    )


st.sidebar.subheader("Dataset")
data = load_data()

if data.empty:
    st.sidebar.warning("Aucune donnée chargée. Veuillez téléverser un fichier.")
else:
    
    st.sidebar.subheader("Paramètres de visualisation")
    
    min_val0 = st.sidebar.slider("valeur minimale de la variable de filtre", 0, 100, 80)
      
    numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
    
    variable_0 = st.sidebar.selectbox("Variable pour filtrer les données", numerical_columns)
    filtered_data = data[data[variable_0] >= min_val0]
    st.write(data.head(3))
    
    st.subheader("Posez une question sur le dataset")
    user_question = st.text_input("Votre question :", placeholder= f"Ex: Quelle est la médiane de {numerical_columns[0]} ?")
    dataset_description = data.describe()  
    first_rows = data.head()  
    dataset_size = data.shape
    dataset_col = data.dtypes.to_dict()
    chart_map =mapp_viz()

    load_dotenv() # à décommenter avant deployement sur streamlit
    api_key = os.getenv('GOOGLE_API_KEY') # à décommenter avant deployement sur streamlit
    
    #api_key = st.secrets["GOOGLE_API_KEY"] # à commenter avant deployement sur streamlit
    
    #CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

   
    def get_gemini_response_with_dataset_info(question):
        
        # Formuler un prompt détaillé pour Claude
        prompt= f"""
        Voici un dataset avec les informations suivantes :
        
        Taille du dataset : {dataset_size}
        Colonnes et types de données : {dataset_col}
        Statistiques descriptives : {dataset_description}
        Exemples de premières lignes : {first_rows}
        Dictionnaire de mapping Chart_map :{chart_map}

        L'utilisateur pose la question suivante : {question}

        Sans justifier ta réponse, Donne la clé du dictionnaire chart_map correspondant le plus à la question
        de l'utilisateur sous le format JSON suivant {{"Question": ["a- *"]}} où a est l'expression correspondante.
        À la fin de cette réponse, liste les colonnes concernées sous le format JSON suivant :
        {{"colonnes_concernees": ["nom_colonne1", "nom_colonne2"]}}
        """
        
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.5,
            max_tokens=300,
            timeout=None,
            max_retries=2,
        )
        
        ai_msg = llm.invoke(prompt)
    
        
        if hasattr(ai_msg, 'content'):
            reponse = ai_msg.content
            return reponse
        else:
            return "Erreur : la réponse générée ne contient pas de texte valide."
        

        
    if user_question:
                
        st.subheader(f"Question : {user_question}")    
        
        response = get_gemini_response_with_dataset_info(user_question)
        
        #st.write(f"Réponse : {response}")
        
        question_cle = extract_question_from_response(response)
        
        if question_cle =="23- Structure du dataset":
            st.write(data.shape)
            st.write(data.describe())
            st.write(data.dtypes.to_dict())
            
        else:
            colonnes_concernees = extract_columns_from_response(response)
            
            chart_type_concernes = chart_map[question_cle]["Graphiques_fct"]
            Mesure_mapp_concernes = chart_map[question_cle]["Calculs_fct"]
            
            
            
            if colonnes_concernees:
                if Mesure_mapp_concernes:
                    colonnes_valides = [col for col in colonnes_concernees if col in data.columns] 
                    if colonnes_valides:
                        st.markdown("### 📊 Statistiques et Visualisation des données")
                        
                        numerical_colonnes_valides = [col for col in numerical_columns if col in colonnes_valides]
                        categorical_colonnes_valides = [col for col in categorical_columns if col in colonnes_valides] 
                        
                        statistic_indic(filtered_data, numerical_colonnes_valides, Mesure_mapp_concernes)
                        visualiz(filtered_data, numerical_colonnes_valides, categorical_colonnes_valides, chart_type_concernes)
                        
                    else:
                        st.warning("❌ Les colonnes identifiées ne sont pas valides dans le dataset.")
            else:
                st.warning("❌ Aucune colonne n'a été identifiée dans la réponse pour plus de visualisation.")
                                    
                                    
                            
                    
            
    


            


           
                    
                    