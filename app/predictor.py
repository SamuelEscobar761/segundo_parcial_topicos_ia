import torch
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentClassifier:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)


    def calcular_valor(self, vector):
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


    def predict(self, text):
        tokens = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**tokens)
        # Aplicar la función de activación sigmoide
        sigmoid = torch.nn.Sigmoid()
        
        probabilities = outputs.logits
        print(probabilities.tolist()[0])
        # Ajustar los valores al rango de -1 a 1
        scaled_value = self.calcular_valor(probabilities.tolist()[0])
        # scaled_value_rounded = round(scaled_value.item(), 3)  # Cambiar 3 por la cantidad de decimales deseada
        return scaled_value
    
    def get_category(self, num):
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
    