import pandas as pd
import streamlit as st


def detect_separator(file):
    
    file.seek(0)  
    first_bytes = file.read(4096).decode('utf-8')  
    file.seek(0)  
    
    if "," in first_bytes:
        return ","
    elif ";" in first_bytes:
        return ";"
    elif "\t" in first_bytes:
        return "\t"
    else:
        return ","  
    
def load_data():
    data = st.sidebar.file_uploader("Téléversez un fichier CSV ou Excel", type=["csv", "xlsx"])
    if data is not None:
        try:
            
            if data.name.endswith(".csv"):
                sep = detect_separator(data)
                data = pd.read_csv(data, sep=sep)
            else:
                data = pd.read_excel(data)
            
            st.success("Fichier chargé avec succès !")
            
            
                    
            return data
        
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            return pd.DataFrame()
    else:
        st.info("Veuillez téléverser un fichier au niveau de la section Dataset pour commencer.")
        
    return pd.DataFrame()