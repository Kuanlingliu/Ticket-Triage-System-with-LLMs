%%writefile 3_prototype_app_improved.py
# --- Complete Prototype Script with Enhanced RAG and Stable JSON Output ---
import streamlit as st
import torch
import json
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import logging
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION & SETUP ---
logging.basicConfig(level=logging.ERROR)

# --- NOTE: drive.mount() should be run in a separate Colab cell BEFORE launching the app ---
# This script assumes Google Drive is already mounted if you are running in Colab.
# from google.colab import drive
# try:
#     drive.mount('/content/drive', force_remount=True)
# except Exception:
#     pass # Fails gracefully if not in Colab

DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Datel_Project"
ASSETS_DIR = os.path.join(DRIVE_PROJECT_PATH, "assets")
MISTRAL_MODEL_PATH = os.path.join(DRIVE_PROJECT_PATH, "saved_models", "Mistral-7B-Instruct-v0.2_optimized_v16")

# --- RESOURCE LOADING (CACHED FOR PERFORMANCE) ---
@st.cache_resource
def load_all_models_and_assets():
    """
    Loads all necessary models and assets for the ENHANCED RAG pipeline.
    ALSO dynamically builds the validation categories from the data.
    """
    assets = {}
    st.info("🔄 Loading all required AI models and assets...")

    # 1. Load ENHANCED RAG Retriever Assets
    try:
        assets['faiss_index'] = faiss.read_index(os.path.join(ASSETS_DIR, "ticket_knowledge.index"))
        ticket_df = pd.read_pickle(os.path.join(ASSETS_DIR, "ticket_knowledge_data.pkl"))

        # Handle NaN values in the 'ISSUE' column by replacing them with 'Unknown'
        ticket_df['ISSUE'] = ticket_df['ISSUE'].fillna('Unknown')

        assets['ticket_data'] = ticket_df
        assets['retrieval_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        st.write("✅ Enhanced Retriever assets loaded.")
    except Exception as e:
        st.error(f"Failed to load retriever assets from {ASSETS_DIR}. Please run the '1_build_assets.py' script first. Error: {e}")
        return None, None

    # 2. Dynamically create the valid categories from the loaded data
    try:
        valid_categories = {
            "ISSUE": sorted(ticket_df['ISSUE'].unique().tolist()), # Sorted for consistency
            "CATEGORY": ["S1000v4", "S1000v3", "S1000", "S1000v4/CRM", "S1000/CRM"],
            "URGENCYCODE": ["1", "3", "4", "5"]
        }
        st.write(f"✅ Dynamically loaded {len(valid_categories['ISSUE'])} unique ISSUE types.")
    except KeyError as e:
        st.error(f"Error: Column '{e}' not found in 'ticket_knowledge_data.pkl'. Cannot create dynamic categories.")
        return None, None

    # 3. Load Mistral (for both Classification and Generation)
    try:
        mistral_base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        mistral_base_model = AutoModelForCausalLM.from_pretrained(
            mistral_base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        assets['mistral_tokenizer'] = AutoTokenizer.from_pretrained(mistral_base_model_name)
        if assets['mistral_tokenizer'].pad_token is None:
            assets['mistral_tokenizer'].pad_token = assets['mistral_tokenizer'].eos_token
        assets['mistral_model'] = PeftModel.from_pretrained(mistral_base_model, MISTRAL_MODEL_PATH)
        assets['mistral_model'].eval()
        st.write("✅ Mistral model loaded.")
    except Exception as e:
        st.error(f"Failed to load Mistral model from {MISTRAL_MODEL_PATH}. Error: {e}")
        return None, None

    st.success("🎉 All AI components are loaded and ready!")
    return assets, valid_categories

# --- IMPROVED JSON PARSING FUNCTIONS ---
def extract_json_from_text(text):
    """
    Robustly extracts a JSON object from a string, trying various patterns.
    """
    # Pattern 1: Code block with ```json
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Pattern 2: Any code block ```
    match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Pattern 3: A standalone JSON object
    match = re.search(r'(\{.*?\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None

def validate_and_fix_classification(classification_dict, valid_categories):
    """
    Validates classification results against the allowed values and fixes them.
    """
    if not isinstance(classification_dict, dict):
        return {"ISSUE": "Unknown", "CATEGORY": "S1000v4", "URGENCYCODE": "3"}

    result = {}
    for field, valid_values in valid_categories.items():
        predicted_value = str(classification_dict.get(field, "")).strip()

        if predicted_value in valid_values:
            result[field] = predicted_value
        else:
            # Fallback to the most common value if validation fails
            if field == "ISSUE":
                result[field] = "Unknown"
            elif field == "CATEGORY":
                result[field] = "S1000v4"
            else: # URGENCYCODE
                result[field] = "3"
    return result

# --- BACKEND LOGIC ---
@torch.no_grad()
def run_mistral_classification(subject, problem, assets, valid_categories, max_retries=3):
    """
    Uses the fine-tuned Mistral model to classify the ticket with improved stability.
    """
    model = assets['mistral_model']
    tokenizer = assets['mistral_tokenizer']

    # Dynamically create the valid values list for the prompt
    issue_list_str = ", ".join(valid_categories['ISSUE'])
    category_list_str = ", ".join(valid_categories['CATEGORY'])
    urgency_list_str = ", ".join(valid_categories['URGENCYCODE'])

    prompt = f"""[INST] You are a technical support ticket classifier. Your task is to analyze the ticket and respond ONLY with a JSON object in the specified format.

Valid values based on historical data:
- ISSUE: {issue_list_str}
- CATEGORY: {category_list_str}
- URGENCYCODE: {urgency_list_str} (1=High, 3=Normal, 5=Low)

Ticket Details:
Subject: {subject}
Problem: {problem}

Respond with ONLY the JSON object in the following format:
```json
{{
  "ISSUE": "value",
  "CATEGORY": "value",
  "URGENCYCODE": "value"
}}
```
[/INST]
"""
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Isolate the response part of the text
            response_text = prediction_text.split("[/INST]")[-1]

            extracted_json = extract_json_from_text(response_text)
            if extracted_json:
                validated_result = validate_and_fix_classification(extracted_json, valid_categories)
                validated_result["raw_output"] = response_text
                validated_result["attempt"] = attempt + 1
                return validated_result

        except Exception as e:
            st.warning(f"Classification attempt {attempt + 1} failed: {str(e)}")
            continue

    # Final fallback if all retries fail
    return {
        "ISSUE": "Unknown",
        "CATEGORY": "S1000v4",
        "URGENCYCODE": "3",
        "error": "Failed to generate valid classification after all attempts.",
        "raw_output": "No valid output generated."
    }

def find_similar_tickets(text, assets, k=5):
    """Uses the vector database to find the top k similar historical tickets."""
    retrieval_model = assets['retrieval_model']
    index = assets['faiss_index']
    ticket_data = assets['ticket_data']
    query_embedding = retrieval_model.encode([text])
    _, indices = index.search(query_embedding, k)
    return ticket_data.iloc[indices[0]]

@torch.no_grad()
def run_mistral_solution_generation(subject, problem, similar_tickets_df, assets):
    """
    Uses Mistral to generate a solution based on multiple historical contexts.
    """
    model = assets['mistral_model']
    tokenizer = assets['mistral_tokenizer']
    combined_context = ""
    num_cases_to_use = min(3, len(similar_tickets_df))

    for i in range(num_cases_to_use):
        ticket = similar_tickets_df.iloc[i]
        case_context = ticket['KNOWLEDGE_TEXT']
        if len(case_context) > 800:
            case_context = case_context[:800] + "..."
        combined_context += f"\n--- HISTORICAL CASE {i+1}: {ticket['SUBJECT']} ---\n{case_context}\n"

    prompt = f"""[INST] You are an expert IT support assistant. Analyze the new ticket below and provide a step-by-step solution by synthesizing insights from the provided historical cases.

MULTIPLE SIMILAR RESOLVED HISTORICAL CASES:
{combined_context}
NEW TICKET TO RESOLVE:
Subject: {subject}
Problem: {problem}

Based on the patterns and solutions from the historical cases, provide a step-by-step solution that combines the best practices from these resolved tickets. Focus on actionable steps. [/INST]
Based on the historical cases, here is a recommended solution:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    with torch.cuda.amp.autocast():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = prediction_text.split("[/INST]")[-1].strip()
    return solution

# --- MAIN TRIAGE PIPELINE ---
def run_full_triage_pipeline(subject, problem, assets, valid_categories):
    """Orchestrates the entire enhanced hybrid RAG process."""
    full_text = f"{subject} | {problem}"
    classification_results = run_mistral_classification(subject, problem, assets, valid_categories)
    similar_tickets_df = find_similar_tickets(full_text, assets, k=5)
    solution_suggestion = run_mistral_solution_generation(subject, problem, similar_tickets_df, assets)
    return {
        "classification": classification_results,
        "suggested_solution": solution_suggestion,
        "relevant_tickets": similar_tickets_df
    }

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Datel AI Triage System")
st.title("🤖 Datel AI Triage System (Enhanced RAG Prototype)")

assets, valid_categories = load_all_models_and_assets()

if assets and valid_categories:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.header("New Ticket Entry")
        subject = st.text_input("Subject", "Example: Batch posting failed")
        problem = st.text_area("Problem Description", "Example: The PI batch 7268 failed to post.", height=200)
        if st.button("Triage Ticket", type="primary", use_container_width=True):
            if subject and problem:
                with st.spinner("🧠 AI is analyzing the ticket..."):
                    st.session_state.triage_result = run_full_triage_pipeline(subject, problem, assets, valid_categories)
            else:
                st.warning("Please enter both a subject and a problem description.")
    with col2:
        st.header("AI Triage Recommendation")
        if 'triage_result' in st.session_state:
            result = st.session_state.triage_result
            st.subheader("1. AI Suggested Classification")
            classification = result['classification']
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Issue Type", classification.get('ISSUE', 'N/A'))
            with col2b:
                st.metric("Category", classification.get('CATEGORY', 'N/A'))
            with col2c:
                st.metric("Urgency", classification.get('URGENCYCODE', 'N/A'))

            if 'error' in classification:
                with st.expander("🔍 Debug Information"):
                    st.warning(f"Classification Issues: {classification['error']}")
                    st.text(f"Raw Output: {classification.get('raw_output', 'N/A')}")
                    st.text(f"Attempts Made: {classification.get('attempt', 'N/A')}")

            st.subheader("2. AI Suggested Solution (Multi-Case Analysis)")
            st.info("💡 This solution combines insights from the top 3 most similar historical cases.")
            st.markdown(result['suggested_solution'])

            st.subheader("3. Top 5 Relevant Historical Tickets")
            st.caption("📊 Cases 1-3 were used for solution generation.")
            display_df = result['relevant_tickets'][['TICKETID', 'SUBJECT', 'KNOWLEDGE_TEXT']].copy()
            display_df['KNOWLEDGE_TEXT'] = display_df['KNOWLEDGE_TEXT'].str[:200] + '...'
            display_df['USED_FOR_SOLUTION'] = ['✓ Used', '✓ Used', '✓ Used', 'Reference', 'Reference']
            st.dataframe(display_df[['TICKETID', 'SUBJECT', 'USED_FOR_SOLUTION', 'KNOWLEDGE_TEXT']], use_container_width=True)

            st.markdown("---")
            st.subheader("Validation & Feedback")
            feedback_cols = st.columns(2)
            with feedback_cols[0]:
                if st.button("👍 Recommendation was helpful", use_container_width=True):
                    st.success("Thank you for your feedback!")
            with feedback_cols[1]:
                if st.button("👎 Recommendation was not helpful", use_container_width=True):
                    st.warning("Thank you for your feedback! We'll use this to improve our system.")
else:
    st.error("Application failed to load. Please check the logs for errors related to model or asset loading.")
