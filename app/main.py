from typing import Union
from fastapi import FastAPI
from predictor import SentimentClassifier  # Importa la clase desde tu predictor.py

app = FastAPI()

# Crea una instancia de SentimentClassifier con las rutas correctas para tu modelo y tokenizer
classifier = SentimentClassifier(model_path="code/", tokenizer_path="code/")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict_sentiment/")  # Nueva ruta para la predicción de sentimientos
def predict_sentiment(text: str):
    predicted_sentiment = classifier.predict(text)  # Utiliza el método predict de la clase SentimentClassifier
    return {"Predicted Sentiment": predicted_sentiment}
