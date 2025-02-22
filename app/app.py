import streamlit as st
import torch
import pickle
import numpy as np
from helpers.utils import BERT, SimpleTokenizer, calculate_similarity

# Load Model & Tokenizer Data
st.title("Sentence Similarity with BERT")

# Load pre-trained model data
data = pickle.load(open("models/bert-pretrained-data.pkl", "rb"))
word2id = data["word2id"]
max_len = data["max_len"]
max_mask = data["max_mask"]

# Initialize Tokenizer
tokenizer = SimpleTokenizer(word2id)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT()

# Fix Missing Keys in Model Weights
model.load_state_dict(torch.load("models/sentence_model.pt", map_location=device), strict=False)
model.to(device)
model.eval()

# User Input
st.write("Enter two sentences to compare their similarity:")
sentence_a = st.text_input("Sentence 1", "Your contribution helped make it possible for us to provide our students with a quality education.")
sentence_b = st.text_input("Sentence 2", "Your contributions were of no help with our students' education.")

# Compute Similarity on Button Click
if st.button("Calculate Similarity"):
    if not sentence_a or not sentence_b:
        st.warning("Please enter both sentences!")
        st.stop()

    # Tokenize Sentences & Check for Errors
    inputs_a = tokenizer.encode(sentence_a)
    inputs_b = tokenizer.encode(sentence_b)

    if not inputs_a["input_ids"] or not inputs_b["input_ids"]:
        st.error("Error: Tokenization failed. Check input text!")
        st.stop()
    
    similarity_score = np.array(calculate_similarity(model, tokenizer, sentence_a, sentence_b, device))
    print("SCORE ###########", similarity_score)
    #st.success(f"**Cosine Similarity Score:** {similarity_score:.4f}")
    #print(f"Score: {similarity_score:.2f}")
    #st.write("Score:", similarity_score)
    st.markdown("Cosine Similarity Score: " + " ".join(similarity_score[0].astype(str)))

# Footer
st.write("Built with ❤️ using Streamlit & BERT")