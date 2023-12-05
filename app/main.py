import os
import spacy
import time
import psutil
import csv
import datetime
from fastapi import FastAPI
from predictor import SentimentClassifier

app = FastAPI()
nlp = spacy.load("es_core_news_sm")
current_directory = os.path.dirname(__file__)
directory_path = os.path.join(current_directory, "code")
classifier = SentimentClassifier(model_path=directory_path, tokenizer_path=directory_path)

# Función para registrar en CSV
def log_to_csv(endpoint, text, response):
    with open('reports.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        now = datetime.datetime.now()
        writer.writerow([now, endpoint, text, response])

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
    predicted_sentiment = classifier.predict(text) 
    response = {"Predicted Sentiment": predicted_sentiment}
    log_to_csv("/sentiment", text, response)
    return response

@app.post("/analysis")
async def analyze_text(text: str):
    start_time = time.time()
    doc = nlp(text)
    end_time = time.time()
    execution_time = end_time - start_time
    sentiment_value = classifier.predict(text)
    sentiment_category = classifier.get_category(sentiment_value)
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
    # Aquí puedes implementar la lógica para descargar el archivo CSV
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
