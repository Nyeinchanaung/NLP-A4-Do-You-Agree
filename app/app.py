import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
from helpers.utils import BERT, SimpleTokenizer, predict_nli_and_similarity

# Load Model & Tokenizer Data
st.title("Sentence Similarity with BERT")

# Load pre-trained model data
data = pickle.load(open("models/bert-pretrained-data.pkl", "rb"))
word2id = data["word2id"]
max_len = data["max_len"]
max_mask = data["max_mask"]
d_model = data["d_model"]  # Needed for classifier head definition

# Initialize Tokenizer
tokenizer = SimpleTokenizer(word2id)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT()

# Load the combined checkpoint
checkpoint = torch.load("models/sentence_classifier_model.pt", map_location=device)

# Load BERT weights
model.load_state_dict(checkpoint["bert"], strict=False)
model.to(device)
model.eval()

# Define and Load Classifier Head
classifier_head = nn.Linear(d_model * 3, 3).to(device)  # 3 classes: Entailment, Neutral, Contradiction
classifier_head.load_state_dict(checkpoint["classifier_head"])
classifier_head.eval()

# User Interface
st.write("Enter a **premise** and a **hypothesis**, and the model will predict the relationship between them.")
premise = st.text_input("Premise", "A man is playing a guitar on stage.")
hypothesis = st.text_input("Hypothesis", "The man is good at music.")

if st.button("Predict"):
    if premise and hypothesis:
        # Call predict_nli_and_similarity with classifier_head
        result = predict_nli_and_similarity(model, premise, hypothesis, device, classifier_head)
        
        # Display results
        st.success(f"**NLI Predicted Label:** {result['nli_label']}")
        st.info(f"**Cosine Similarity Score:** {result['similarity_score']:.4f}")

    else:
        st.warning("Please enter both a premise and a hypothesis!")

# Footer
st.write("Built with ❤️ using Streamlit & BERT")