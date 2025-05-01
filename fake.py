import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(text):
    if not isinstance(text, str):
        return ""

    try:
        text = text.lower()
        text = re.sub(r"http\S+|[^a-z\s]", "", text)
        tokens = nltk.word_tokenize(text)  # make sure this is correct!
        tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error in preprocessing: {text}\n{e}")
        return ""


def predict_label(text):
    model.eval()
    text = preprocess(text)  # same cleaning as training
    encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prediction = torch.argmax( output, dim=1).item()

   
    return  prediction

import torch.nn.functional as f
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('roberta_tokenizer')

# Define the model class exactly as before
class RobertaWithFFNN(nn.Module):
    def _init_(self):
        super(RobertaWithFFNN, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.3)
        self.ffnn = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        return self.ffnn(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaWithFFNN().to(device)
# Load full model
model = torch.load("model/roberta_fakenews_ffnn.pth", map_location=torch.device("cpu"))
model.eval()


import streamlit as st
from streamlit_option_menu import option_menu
st.title("Fake News Detection App")
with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu",
        options=["Home","About App","About Us"],
        menu_icon="cast",
        icons=["house","book","people"],
       
    )

if selected=="Home":
    news=st.text_input("input news statements")
    if st.button("predict"):
        pred=predict_label(news)
        if pred==0:
            st.info("The News is Fake")
        else:
            st.info("The News is Real")