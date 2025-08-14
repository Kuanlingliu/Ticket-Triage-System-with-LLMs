!pip install streamlit

%%writefile 3_prototype_app.py
# --- Complete Prototype Script for Action Plan Step 2 ---
import streamlit as st
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
import logging
import os
from google.colab import drive
import joblib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION & SETUP ---
# Configure logging to be clean
logging.basicConfig(level=logging.ERROR)

# Mount Google Drive & Define Paths
try:
    drive.mount('/content/drive', force_remount=True)
except Exception as e:
    print(f"Drive already mounted or error mounting: {e}")

DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Datel_Project"
ASSETS_DIR = os.path.join(DRIVE_PROJECT_PATH, "assets")
GEMMA_MODEL_PATH = os.path.join(DRIVE_PROJECT_PATH, "saved_models", "gemma-2b-it_optimized_v15")
BERT_MODEL_PATH = os.path.join(DRIVE_PROJECT_PATH, "saved_models", "bert_model_v2", "bert-base-uncased_triage_v16")


# --- RESOURCE LOADING (CACHED FOR PERFORMANCE) ---

@st.cache_resource
def load_all_models_and_assets():
    """
    Loads all necessary models and assets for the RAG pipeline.
    This includes Gemma, BERT, and the FAISS vector database.
    """
    assets = {}
    st.info("🔄 Loading all required AI models and assets...")

    # 1. Load RAG Retriever Assets
    try:
        assets['faiss_index'] = faiss.read_index(os.path.join(ASSETS_DIR, "ticket_solutions.index"))
        assets['ticket_data'] = pd.read_pickle(os.path.join(ASSETS_DIR, "ticket_data.pkl"))
        assets['retrieval_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        st.write("✅ Retriever assets loaded.")
    except Exception as e:
        st.error(f"Failed to load retriever assets from {ASSETS_DIR}. Please run '1_build_assets.py' first. Error: {e}")
        return None

    # 2. Load Gemma (for Generation)
    try:
        gemma_base_model_name = "google/gemma-2b-it"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        gemma_base_model = AutoModelForCausalLM.from_pretrained(gemma_base_model_name, quantization_config=bnb_config, device_map="auto")
        assets['gemma_tokenizer'] = AutoTokenizer.from_pretrained(gemma_base_model_name)
        assets['gemma_model'] = PeftModel.from_pretrained(gemma_base_model, GEMMA_MODEL_PATH)
        assets['gemma_model'].eval()
        st.write("✅ Gemma model loaded.")
    except Exception as e:
        st.error(f"Failed to load Gemma model from {GEMMA_MODEL_PATH}. Please run the Gemma training script. Error: {e}")
        return None

    # 3. Load BERT (for Classification)
    try:
        assets['bert_model'] = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        assets['bert_tokenizer'] = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        assets['mlb'] = joblib.load(os.path.join(BERT_MODEL_PATH, "mlb_binarizer.joblib"))
        st.write("✅ BERT model loaded.")
    except Exception as e:
        st.error(f"Failed to load BERT model from {BERT_MODEL_PATH}. Please run the BERT training script. Error: {e}")
        return None

    st.success("🎉 All AI components are loaded and ready!")
    return assets

# --- BACKEND LOGIC (Action 2.1) ---

@torch.no_grad()
def run_bert_classification(subject, problem, assets):
    """Uses the highly accurate BERT model to classify the ticket."""
    model = assets['bert_model']
    tokenizer = assets['bert_tokenizer']
    mlb = assets['mlb']

    full_text = f"{subject} | {problem}"
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
    outputs = model(**inputs)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs > 0.5)] = 1

    decoded_labels = mlb.inverse_transform(predictions.reshape(1, -1))[0]

    result = {}
    for label in decoded_labels:
        key, value = label.split(":", 1)
        result[key] = value

    return {
        "ISSUE": result.get("ISSUE", "N/A"),
        "CATEGORY": result.get("CATEGORY", "N/A"),
        "URGENCYCODE": result.get("URGENCYCODE", "N/A")
    }

def find_similar_tickets(text, assets, k=5):
    """Uses the vector database to find the top k similar historical tickets."""
    retrieval_model = assets['retrieval_model']
    index = assets['faiss_index']
    ticket_data = assets['ticket_data']

    query_embedding = retrieval_model.encode([text])
    distances, indices = index.search(query_embedding, k)

    # Return the actual data for the found tickets
    return ticket_data.iloc[indices[0]]

@torch.no_grad()
def run_gemma_solution_generation(subject, problem, context_solution, assets):
    """Uses the flexible Gemma model to generate a human-readable solution."""
    model = assets['gemma_model']
    tokenizer = assets['gemma_tokenizer']

    prompt = f"""### INSTRUCTION
You are an expert AI support assistant. Your task is to analyze a new support ticket and, using the provided historical context, suggest a potential solution.

### CONTEXT FROM A SIMILAR HISTORICAL TICKET
{context_solution}

### NEW TICKET TO ANALYZE
Subject: {subject}
Problem: {problem}

### SUGGESTED SOLUTION
Based on the historical context, here is a suggested solution:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up the output to only show the generated part
    return prediction_text.split("### SUGGESTED SOLUTION")[-1].strip()

# --- MAIN TRIAGE PIPELINE ---

def run_full_triage_pipeline(subject, problem, assets):
    """Orchestrates the entire hybrid RAG process."""
    full_text = f"{subject} | {problem}"

    # 1. Use BERT for fast and accurate classification
    classification_results = run_bert_classification(subject, problem, assets)

    # 2. Use Retriever to find top 5 similar tickets
    similar_tickets_df = find_similar_tickets(full_text, assets, k=5)

    # 3. Use Gemma to generate a solution based on the #1 most similar ticket
    top_context_solution = similar_tickets_df.iloc[0]['SOLUTION']
    solution_suggestion = run_gemma_solution_generation(subject, problem, top_context_solution, assets)

    return {
        "classification": classification_results,
        "suggested_solution": solution_suggestion,
        "relevant_tickets": similar_tickets_df
    }

# --- STREAMLIT UI (Action 2.2) ---

st.set_page_config(layout="wide", page_title="Datel AI Triage System")
st.title("🤖 Datel AI Triage System (Hybrid RAG Prototype)")

# Load all assets and models at the start
assets = load_all_models_and_assets()

if assets:
    col1, col2 = st.columns([1, 1.5]) # Make the right column wider

    with col1:
        st.header("New Ticket Entry")

        subject = st.text_input("Subject", "Example: Batch posting failed")
        problem = st.text_area("Problem Description", "Example: The PI batch 7268 failed to post and I cannot find the error. The system shows a 'period closed' error but our accounting period is definitely open.", height=200)

        if st.button("Triage Ticket", type="primary", use_container_width=True):
            if subject and problem:
                with st.spinner("🧠 AI is analyzing the ticket... (This may take a moment)"):
                    st.session_state.triage_result = run_full_triage_pipeline(subject, problem, assets)
            else:
                st.warning("Please enter both a subject and a problem description.")

    with col2:
        st.header("AI Triage Recommendation")
        if 'triage_result' in st.session_state:
            result = st.session_state.triage_result

            # --- Output 1: AI Suggested Classification (from BERT) ---
            st.subheader("1. AI Suggested Classification (from BERT)")
            st.json(result['classification'])

            # --- Output 2: AI Suggested Solution (from Gemma + RAG) ---
            st.subheader("2. AI Suggested Solution (from Gemma + RAG)")
            st.markdown(result['suggested_solution'])

            # --- Output 3: Relevant Historical Tickets (from Retriever) ---
            st.subheader("3. Top 5 Relevant Historical Tickets")
            st.dataframe(result['relevant_tickets'][['TICKETID', 'SUBJECT', 'SOLUTION']], use_container_width=True)

            # --- Feedback Mechanism (Task 9) ---
            st.markdown("---")
            st.subheader("Validation & Feedback")
            if st.button("👍 Recommendation was helpful"):
                st.success("Thank you for your feedback!")
                # In a real system, this would log the positive feedback
            if st.button("👎 Recommendation was not helpful"):
                st.warning("Thank you for your feedback! This will help us improve.")
                # In a real system, this would log the negative feedback
