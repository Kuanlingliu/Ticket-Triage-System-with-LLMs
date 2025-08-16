%%writefile 3_prototype_app_improved.py
# --- Complete Prototype Script with Enhanced RAG and Stable JSON Output ---
import streamlit as st
import torch
import json
import pandas as pd
import re
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
logging.basicConfig(level=logging.ERROR)
# --- FIX: drive.mount() should be run in a separate Colab cell BEFORE launching the app ---
# This line is removed from the script itself.
# drive.mount('/content/drive', force_remount=True)

DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Datel_Project"
ASSETS_DIR = os.path.join(DRIVE_PROJECT_PATH, "assets")
# --- NOTE: Using Mistral as the best performing model from your tests ---
MISTRAL_MODEL_PATH = os.path.join(DRIVE_PROJECT_PATH, "saved_models", "Mistral-7B-Instruct-v0.2_optimized_v16")

# --- PREDEFINED CATEGORIES FOR VALIDATION (Based on Actual Data Distribution) ---
VALID_CATEGORIES = {
    "ISSUE": [
        "Unknown", "Inventory", "Printing", "Cashbook", "MTD Query", 
        "Reports", "Performance", "MS", "Paperless", "EDI",
        # Add more based on your full dataset - these are just the top 10
        "Integration", "Database", "User Access", "Configuration", 
        "Email", "Network", "Hardware", "Other"
    ],
    "CATEGORY": ["S1000v4", "S1000v3", "S1000", "S1000v4/CRM", "S1000/CRM"],
    "URGENCYCODE": ["1", "3", "4", "5"]  # Based on your data: 1=High, 3=Normal, 4=?, 5=Low
}

# --- RESOURCE LOADING (CACHED FOR PERFORMANCE) ---

@st.cache_resource
def load_all_models_and_assets():
    """
    Loads all necessary models and assets for the ENHANCED RAG pipeline.
    """
    assets = {}
    st.info("🔄 Loading all required AI models and assets...")

    # 1. Load ENHANCED RAG Retriever Assets
    try:
        # --- CHANGE: Load the new, enhanced knowledge assets ---
        assets['faiss_index'] = faiss.read_index(os.path.join(ASSETS_DIR, "ticket_knowledge.index"))
        assets['ticket_data'] = pd.read_pickle(os.path.join(ASSETS_DIR, "ticket_knowledge_data.pkl"))
        assets['retrieval_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        st.write("✅ Enhanced Retriever assets loaded.")
    except Exception as e:
        st.error(f"Failed to load retriever assets from {ASSETS_DIR}. Please run the updated '1_build_assets.py' first. Error: {e}")
        return None

    # 2. Load Mistral (for both Classification and Generation)
    try:
        mistral_base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True  # Improved quantization stability
        )
        mistral_base_model = AutoModelForCausalLM.from_pretrained(
            mistral_base_model_name, 
            quantization_config=bnb_config, 
            device_map="auto",
            torch_dtype=torch.bfloat16  # Consistent dtype
        )
        assets['mistral_tokenizer'] = AutoTokenizer.from_pretrained(mistral_base_model_name)
        
        # Ensure pad token is set
        if assets['mistral_tokenizer'].pad_token is None:
            assets['mistral_tokenizer'].pad_token = assets['mistral_tokenizer'].eos_token
            
        assets['mistral_model'] = PeftModel.from_pretrained(mistral_base_model, MISTRAL_MODEL_PATH)
        assets['mistral_model'].eval()
        st.write("✅ Mistral model loaded.")
    except Exception as e:
        st.error(f"Failed to load Mistral model from {MISTRAL_MODEL_PATH}. Error: {e}")
        return None
        
    st.success("🎉 All AI components are loaded and ready!")
    return assets

# --- IMPROVED JSON PARSING FUNCTIONS ---

def extract_json_from_text(text):
    """
    Robust JSON extraction from model output with focus on the exact expected format.
    """
    # Strategy 1: Look for the exact format with triple backticks and space
    exact_pattern = r'```json\s*\{\s*"ISSUE":\s*"([^"]+)",\s*"CATEGORY":\s*"([^"]+)",\s*"URGENCYCODE":\s*"([^"]+)"\s*\}\s*```'
    match = re.search(exact_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return {
                "ISSUE": match.group(1).strip(),
                "CATEGORY": match.group(2).strip(), 
                "URGENCYCODE": match.group(3).strip()
            }
        except:
            pass
    
    # Strategy 2: Look for ```json blocks with any valid JSON inside
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                if isinstance(parsed, dict) and all(key in parsed for key in ["ISSUE", "CATEGORY", "URGENCYCODE"]):
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Look for standalone JSON objects
    json_pattern = r'\{[^{}]*"ISSUE"[^{}]*"CATEGORY"[^{}]*"URGENCYCODE"[^{}]*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                if isinstance(parsed, dict) and all(key in parsed for key in ["ISSUE", "CATEGORY", "URGENCYCODE"]):
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Strategy 4: Extract key-value pairs manually
    issue_match = re.search(r'"?ISSUE"?\s*:\s*"?([^",\n}]+)"?', text, re.IGNORECASE)
    category_match = re.search(r'"?CATEGORY"?\s*:\s*"?([^",\n}]+)"?', text, re.IGNORECASE)
    urgency_match = re.search(r'"?URGENCYCODE"?\s*:\s*"?([1345]|Low|Medium|High|Critical|Normal)"?', text, re.IGNORECASE)
    
    if issue_match or category_match or urgency_match:
        urgency_value = "3"  # Default
        if urgency_match:
            raw_urgency = urgency_match.group(1).strip().lower()
            # Map text urgency to numbers
            urgency_mapping = {"low": "5", "normal": "3", "medium": "3", "high": "1", "critical": "1"}
            urgency_value = urgency_mapping.get(raw_urgency, raw_urgency)
            
        return {
            "ISSUE": issue_match.group(1).strip() if issue_match else "Unknown",
            "CATEGORY": category_match.group(1).strip() if category_match else "S1000v4",
            "URGENCYCODE": urgency_value
        }
    
    return None

def validate_and_fix_classification(classification_dict):
    """
    Validates classification results and fixes invalid values with closest matches.
    """
    if not isinstance(classification_dict, dict):
        return {"ISSUE": "Unknown", "CATEGORY": "S1000v4", "URGENCYCODE": "3"}
    
    result = {}
    
    for field, valid_values in VALID_CATEGORIES.items():
        predicted_value = classification_dict.get(field, "")
        
        # Clean the predicted value
        predicted_value = str(predicted_value).strip()
        
        # For URGENCYCODE, keep as string numbers, for others use title case
        if field != "URGENCYCODE":
            predicted_value = predicted_value.title()
        
        # Exact match
        if predicted_value in valid_values:
            result[field] = predicted_value
        else:
            # Find closest match using simple string similarity
            if field == "ISSUE":
                best_match = "Unknown"  # Most common category
            elif field == "CATEGORY":
                best_match = "S1000v4"  # Most common category  
            else:  # URGENCYCODE
                best_match = "3"  # Most common urgency code
            
            best_score = 0
            
            for valid_value in valid_values:
                # Simple similarity: check if predicted value is contained in valid value or vice versa
                similarity = 0
                pred_lower = predicted_value.lower()
                valid_lower = valid_value.lower()
                
                if pred_lower in valid_lower or valid_lower in pred_lower:
                    similarity = min(len(predicted_value), len(valid_value)) / max(len(predicted_value), len(valid_value))
                
                # Special handling for common variations
                if field == "URGENCYCODE":
                    # Map common urgency terms to numbers
                    urgency_mapping = {
                        "low": "5", "normal": "3", "medium": "3", 
                        "high": "1", "critical": "1", "urgent": "1"
                    }
                    if pred_lower in urgency_mapping:
                        if urgency_mapping[pred_lower] == valid_value:
                            similarity = 1.0
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = valid_value
            
            result[field] = best_match
    
    return result

# --- BACKEND LOGIC ---

@torch.no_grad()
def run_mistral_classification(subject, problem, assets, max_retries=3):
    """
    Uses the fine-tuned Mistral model to classify the ticket with improved stability.
    """
    model = assets['mistral_model']
    tokenizer = assets['mistral_tokenizer']
    
    # Improved prompt with exact format specification
    prompt = f"""[INST] You are a technical support ticket classifier for a business software system. Analyze the ticket and respond with EXACTLY this format including the triple backticks.

Valid values based on historical data:
- ISSUE: Unknown, Inventory, Printing, Cashbook, MTD Query, Reports, Performance, MS, Paperless, EDI, Integration, Database, User Access, Configuration, Email, Network, Hardware, Other
- CATEGORY: S1000v4, S1000v3, S1000, S1000v4/CRM, S1000/CRM
- URGENCYCODE: 1 (High/Critical), 3 (Normal), 4, 5 (Low)

Ticket Details:
Subject: {subject}
Problem: {problem}

Respond with EXACTLY this format (copy this format exactly):

```json
 {{ "ISSUE": "value", "CATEGORY": "value", "URGENCYCODE": "value" }} ```
```

Replace "value" with appropriate classifications. Use EXACTLY this format including the backticks and spacing. [/INST]

```json
 {{"""
    
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,
                padding=True
            ).to(model.device)
            
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=150,
                    min_new_tokens=20,
                    do_sample=False,  # Deterministic output
                    temperature=0.1,  # Low temperature for consistency
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][len(inputs['input_ids'][0]):]
            prediction_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Extract and validate JSON
            extracted_json = extract_json_from_text(prediction_text)
            if extracted_json:
                validated_result = validate_and_fix_classification(extracted_json)
                validated_result["raw_output"] = prediction_text
                validated_result["attempt"] = attempt + 1
                return validated_result
                
        except Exception as e:
            st.warning(f"Classification attempt {attempt + 1} failed: {str(e)}")
            continue
    
    # Final fallback - return default classification based on most common values
    return {
        "ISSUE": "Unknown",        # Most common ISSUE (3704 occurrences)
        "CATEGORY": "S1000v4",     # Most common CATEGORY (4121 occurrences) 
        "URGENCYCODE": "3",        # Most common URGENCYCODE (4556 occurrences)
        "error": "Failed to generate valid classification after all attempts",
        "raw_output": "No valid output generated"
    }

def find_similar_tickets(text, assets, k=5):
    """Uses the vector database to find the top k similar historical tickets."""
    retrieval_model = assets['retrieval_model']
    index = assets['faiss_index']
    ticket_data = assets['ticket_data']
    
    query_embedding = retrieval_model.encode([text])
    distances, indices = index.search(query_embedding, k)
    
    return ticket_data.iloc[indices[0]]

@torch.no_grad()
def run_mistral_solution_generation(subject, problem, similar_tickets_df, assets):
    """
    Uses Mistral to generate a solution based on multiple historical contexts.
    Enhanced to utilize top 3 most similar cases for more comprehensive solutions.
    """
    model = assets['mistral_model']
    tokenizer = assets['mistral_tokenizer']
    
    # Combine knowledge from top 3 most similar tickets for richer context
    combined_context = ""
    num_cases_to_use = min(3, len(similar_tickets_df))
    
    for i in range(num_cases_to_use):
        ticket = similar_tickets_df.iloc[i]
        case_context = ticket['KNOWLEDGE_TEXT']
        
        # Truncate each case to prevent context overflow while keeping essential information
        if len(case_context) > 800:
            case_context = case_context[:800] + "..."
        
        combined_context += f"\n--- HISTORICAL CASE {i+1}: {ticket['SUBJECT']} (ID: {ticket['TICKETID']}) ---\n"
        combined_context += case_context
        combined_context += "\n" + "="*50 + "\n"
    
    prompt = f"""[INST] You are an expert IT support assistant. Analyze the new ticket below and provide a comprehensive solution by synthesizing insights from multiple similar historical cases.

MULTIPLE SIMILAR RESOLVED HISTORICAL CASES:
{combined_context}

NEW TICKET TO RESOLVE:
Subject: {subject}
Problem: {problem}

Based on the patterns and solutions from the historical cases above, provide a step-by-step solution that combines the best practices from these resolved tickets. Focus on actionable steps that have proven successful in similar situations. [/INST]

Based on analyzing the historical cases, here is a comprehensive solution combining proven approaches:

"""
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=3072,  # Increased context length to accommodate multiple cases
        padding=True
    ).to(model.device)
    
    with torch.cuda.amp.autocast():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=400,  # Increased from 350 to prevent truncation
            min_new_tokens=50,   # Ensure minimum completion length
            do_sample=True, 
            temperature=0.7, 
            top_p=0.9, 
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=False  # Prevent premature stopping
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][len(inputs['input_ids'][0]):]
    prediction_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Post-processing to handle potential truncation
    prediction_text = prediction_text.strip()
    
    # If text ends mid-sentence, try to complete it gracefully
    if prediction_text and not prediction_text.endswith(('.', '!', '?', ':')):
        # Find the last complete sentence
        sentences = prediction_text.split('.')
        if len(sentences) > 1:
            # Keep all complete sentences, discard the incomplete last part
            prediction_text = '.'.join(sentences[:-1]) + '.'
    
    return prediction_text

# --- MAIN TRIAGE PIPELINE ---

def run_full_triage_pipeline(subject, problem, assets):
    """Orchestrates the entire ENHANCED hybrid RAG process with multi-case solution generation."""
    full_text = f"{subject} | {problem}"
    
    classification_results = run_mistral_classification(subject, problem, assets)
    similar_tickets_df = find_similar_tickets(full_text, assets, k=5)
    
    # ENHANCED: Pass the entire similar_tickets_df to utilize multiple cases
    solution_suggestion = run_mistral_solution_generation(subject, problem, similar_tickets_df, assets)
    
    return {
        "classification": classification_results,
        "suggested_solution": solution_suggestion,
        "relevant_tickets": similar_tickets_df
    }

# --- STREAMLIT UI ---

st.set_page_config(layout="wide", page_title="Datel AI Triage System")
st.title("🤖 Datel AI Triage System (Enhanced RAG Prototype)")

assets = load_all_models_and_assets()

if assets:
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.header("New Ticket Entry")
        subject = st.text_input("Subject", "Example: Batch posting failed")
        problem = st.text_area("Problem Description", "Example: The PI batch 7268 failed to post.", height=200)
        
        if st.button("Triage Ticket", type="primary", use_container_width=True):
            if subject and problem:
                with st.spinner("🧠 AI is analyzing the ticket..."):
                    st.session_state.triage_result = run_full_triage_pipeline(subject, problem, assets)
            else:
                st.warning("Please enter both a subject and a problem description.")

    with col2:
        st.header("AI Triage Recommendation")
        if 'triage_result' in st.session_state:
            result = st.session_state.triage_result
            
            st.subheader("1. AI Suggested Classification (from Mistral)")
            classification = result['classification']
            
            # Display classification in a more user-friendly way
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Issue Type", classification.get('ISSUE', 'N/A'))
            with col2b:
                st.metric("Category", classification.get('CATEGORY', 'N/A'))
            with col2c:
                st.metric("Urgency", classification.get('URGENCYCODE', 'N/A'))
            
            # Show debug info if there were issues
            if 'error' in classification:
                with st.expander("🔍 Debug Information"):
                    st.warning(f"Classification Issues: {classification['error']}")
                    st.text(f"Raw Output: {classification.get('raw_output', 'N/A')}")
                    st.text(f"Attempts Made: {classification.get('attempt', 'N/A')}")
            
            st.subheader("2. AI Suggested Solution (Enhanced Multi-Case Analysis)")
            solution_info = st.info("💡 This solution combines insights from the top 3 most similar historical cases")
            st.markdown(result['suggested_solution'])
            
            st.subheader("3. Top 5 Relevant Historical Tickets")
            st.caption("📊 Cases 1-3 were used for solution generation, Cases 4-5 provided for additional reference")
            # Display the tickets with indicators for which were used in solution generation
            display_df = result['relevant_tickets'][['TICKETID', 'SUBJECT', 'KNOWLEDGE_TEXT']].copy()
            # Truncate KNOWLEDGE_TEXT for display
            display_df['KNOWLEDGE_TEXT'] = display_df['KNOWLEDGE_TEXT'].str[:200] + '...'
            
            # Add usage indicator
            display_df['USED_FOR_SOLUTION'] = ['✓ Used', '✓ Used', '✓ Used', 'Reference', 'Reference']
            
            st.dataframe(display_df[['TICKETID', 'SUBJECT', 'USED_FOR_SOLUTION', 'KNOWLEDGE_TEXT']], use_container_width=True)
            
            st.markdown("---")
            st.subheader("Validation & Feedback")
            col_feedback1, col_feedback2 = st.columns(2)
            with col_feedback1:
                if st.button("👍 Recommendation was helpful", use_container_width=True):
                    st.success("Thank you for your feedback!")
            with col_feedback2:
                if st.button("👎 Recommendation was not helpful", use_container_width=True):
                    st.warning("Thank you for your feedback! We'll work to improve our recommendations.")
