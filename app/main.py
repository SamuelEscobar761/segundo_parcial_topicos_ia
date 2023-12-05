import os
import spacy
import time
import psutil
from typing import Union
from fastapi import FastAPI
from predictor import SentimentClassifier  # Importa la clase desde tu predictor.py

app = FastAPI()
nlp = spacy.load("es_core_news_sm")


# Get the current directory of the script
current_directory = os.path.dirname(__file__)

# Relative path to the directory containing your model and tokenizer files
directory_path = os.path.join(current_directory, "code")

# Crea una instancia de SentimentClassifier con las rutas correctas para tu modelo y tokenizer
classifier = SentimentClassifier(model_path=directory_path, tokenizer_path=directory_path)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/sentiment/")  # Nueva ruta para la predicción de sentimientos
def predict_sentiment(text: str):
    predicted_sentiment = classifier.predict(text)  # Utiliza el método predict de la clase SentimentClassifier

    return {"Predicted Sentiment": predicted_sentiment}


@app.post("/analysis")
async def analyze_text(text: str):
    start_time = time.time()  # Tiempo de inicio de la ejecución

    # Realizar análisis de texto usando SpaCy
    doc = nlp(text)

    end_time = time.time()  # Tiempo de finalización de la ejecución
    execution_time = end_time - start_time  # Tiempo total de ejecución

    # Obtener análisis de sentimiento (solo un ejemplo, puedes usar otra librería o método)
    sentiment_value = classifier.predict(text)  # Esto sería el valor de sentimiento calculado
    sentiment_category = classifier.get_category(sentiment_value)  # Aquí se asignaría la categoría de sentimiento

    # Calcular tamaño del texto procesado (carácteres)
    text_size = len(text)

    # Calcular la memoria utilizada por el proceso
    process = psutil.Process()
    memory_used = process.memory_info().rss  # Memoria utilizada en bytes

    # Recopilar información de la predicción
    prediction_info = {
        "POS_tags": [token.pos_ for token in doc],
        "NER": [(ent.text, ent.label_) for ent in doc.ents],
        "Embedding": [token.vector.tolist() for token in doc],
        "Sentiment_Value": sentiment_value,
        "Sentiment_Category": sentiment_category
    }

    # Información de la ejecución de la predicción
    execution_info = {
        "Execution_Time": execution_time,  # Tiempo total de ejecución en segundos
        "Text_Size": text_size,  # Tamaño del texto procesado, número de carácteres
        "Memory_Used": memory_used  # Memoria utilizada por el proceso en bytes
    }

    # Combinar información de la predicción y ejecución
    response = {
        "Prediction_Info": prediction_info,
        "Execution_Info": execution_info
    }

    return response



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)