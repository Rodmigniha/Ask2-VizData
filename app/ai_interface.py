import json
import re

def extract_columns_from_response(response_text):
    try:
        # Trouver la partie JSON dans la réponse
        json_match = re.search(r'{"colonnes_concernees": \[[^\]]*\]}', response_text, re.DOTALL)
        if json_match:
            json_part = json_match.group(0)
            json_part=json.loads(json_part)              
            return json_part["colonnes_concernees"]
    except Exception as e:
        return []
            
def extract_question_from_response(response_text):
    try:
        # Trouver la partie JSON dans la réponse 
        json_match = re.search(r'{"Question": \[[^\]]*\]}', response_text, re.DOTALL)
        if json_match:
            json_part = json_match.group(0)
            json_part=json.loads(json_part)              
            return json_part["Question"][0]
    except Exception as e:
        return []
    
def extract_chart_from_response(response_text):
    try:
        # Trouver la partie JSON dans la réponse 
        json_match = re.search(r'{"chart_type": \[[^\]]*\]}', response_text, re.DOTALL)
        if json_match:
            json_part = json_match.group(0)
            json_part=json.loads(json_part)              
            return json_part["chart_type"]
    except Exception as e:
        return []
