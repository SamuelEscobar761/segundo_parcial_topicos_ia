import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentClassifier:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.label_mapping = {
            0: "1 star",
            1: "2 stars",
            2: "3 stars",
            3: "4 stars",
            4: "5 stars"
        }

    def predict(self, text):
        tokens = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**tokens)
        predicted_class = torch.argmax(outputs.logits)
        predicted_label = self.label_mapping.get(predicted_class.item(), "Desconocido")
        return predicted_label