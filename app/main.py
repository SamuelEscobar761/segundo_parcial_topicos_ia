import os
import spacy
import time
import psutil
import csv
import datetime
import torch
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()
nlp = spacy.load("es_core_news_sm")
current_directory = os.path.dirname(__file__)
directory_path = os.path.join(current_directory, "code")

# Función para registrar en CSV
def log_to_csv(endpoint, text, response):
    file_exists = os.path.isfile('reports.csv')  # Comprueba si el archivo ya existe
    with open('reports.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Si el archivo no existe, escribe los encabezados
            writer.writerow(['Timestamp', 'Endpoint', 'Input Text', 'Response'])
        now = datetime.datetime.now()
        writer.writerow([now, endpoint, text, response])

def predict(text, model_path, tokenizer_path):
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokens = tokenizer(text, return_tensors='pt')
        outputs = model(**tokens)
        # Aplicar la función de activación sigmoide
        sigmoid = torch.nn.Sigmoid()
        
        probabilities = outputs.logits
        print(probabilities.tolist()[0])
        # Ajustar los valores al rango de -1 a 1
        scaled_value = calcular_valor(probabilities.tolist()[0])
        # scaled_value_rounded = round(scaled_value.item(), 3)  # Cambiar 3 por la cantidad de decimales deseada
        return scaled_value

def get_category(num):
        if(num == 1):
            return "muy positivo"
        elif(num == -1):
            return "muy negativo"
        elif(num == 0):
            return "neutro neto"
        elif(num > 0.5):
            return "positivo"
        elif(num > 0):
            return "neutro positivo"
        elif(num > -0.5):
            return "neutro negativo"
        elif(num > -1):
            return "negativo"
        
def calcular_valor(vector):
        vector[0] = vector[0]*-1
        vector[1] = vector[1]*-0.5
        vector[2] = 0
        vector[3] = vector[3]*0.5
        numeroTotal = sum(vector)/len(vector)
        if(numeroTotal > 1):
            numeroTotal = 1
        elif(numeroTotal < -1):
            numeroTotal = -1
        return numeroTotal

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/status")
def get_status():
    status_info = {
        "service_status": "running",
        "system_info": {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent
        },
        "model_info": {
            "sentiment_model": "AutoModelForSequenceClassification",
            "nlp_model": "es_core_news_sm"
        }
    }
    return status_info

@app.post("/sentiment/")  
def predict_sentiment(text: str):
    predicted_sentiment = predict(text, directory_path, directory_path) 
    response = {"Predicted Sentiment": predicted_sentiment}
    log_to_csv("/sentiment", text, response)
    return response

@app.post("/analysis")
async def analyze_text(text: str):
    start_time = time.time()
    doc = nlp(text)
    end_time = time.time()
    execution_time = end_time - start_time
    sentiment_value = predict(text)
    sentiment_category = get_category(sentiment_value)
    text_size = len(text)
    process = psutil.Process()
    memory_used = process.memory_info().rss
    prediction_info = {
        "POS_tags": [token.pos_ for token in doc],
        "NER": [(ent.text, ent.label_) for ent in doc.ents],
        "Embedding": [token.vector.tolist() for token in doc],
        "Sentiment_Value": sentiment_value,
        "Sentiment_Category": sentiment_category
    }
    execution_info = {
        "Execution_Time": execution_time,
        "Text_Size": text_size,
        "Memory_Used": memory_used
    }
    response = {
        "Prediction_Info": prediction_info,
        "Execution_Info": execution_info
    }
    log_to_csv("/analysis", text, response)
    return response

@app.get("/reports")
def get_reports():
    file_path = 'reports.csv'
    return FileResponse(file_path, media_type='text/csv', filename='reports.csv')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
