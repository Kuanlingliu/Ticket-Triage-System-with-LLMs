# --- redesign prompt engineer ---
def create_enhanced_prompt(row):
    subject = str(row['SUBJECT']).strip()
    problem = str(row['PROBLEM']).strip()

    # make sure the format is the same
    issue_value = str(row['ISSUE']).strip() if str(row['ISSUE']).strip() != 'nan' else 'Unknown'
    category_value = str(row['CATEGORY']).strip()
    urgency_value = str(row['URGENCYCODE']).strip()

    return f"""### INSTRUCTION
You are a ticket classifier. You MUST output EXACTLY this JSON format with these EXACT field names:

### REQUIRED FORMAT
```json
{{
  "ISSUE": "value",
  "CATEGORY": "value",
  "URGENCYCODE": "value"
}}
```

### VALID VALUES
ISSUE: Reports, Inventory, Access, Database, Configuration, Other
CATEGORY: S1000, S1000v3, S1000v4, S1000v5, CRM, Other
URGENCYCODE: 1, 2, 3, 4

### EXAMPLES
Password issue → {{"ISSUE": "Access", "CATEGORY": "S1000v3", "URGENCYCODE": "3"}}
Report problem → {{"ISSUE": "Reports", "CATEGORY": "S1000", "URGENCYCODE": "2"}}
Database query → {{"ISSUE": "Database", "CATEGORY": "S1000v4", "URGENCYCODE": "1"}}

### TICKET
Subject: {subject}
Problem: {problem}

### OUTPUT
```json
{{
  "ISSUE": "{issue_value}",
  "CATEGORY": "{category_value}",
  "URGENCYCODE": "{urgency_value}"
}}
```"""

# --- Stricter training version ---
def create_strict_format_prompt(row):
    subject = str(row['SUBJECT']).strip()
    problem = str(row['PROBLEM']).strip()

    issue_value = str(row['ISSUE']).strip() if str(row['ISSUE']).strip() != 'nan' else 'Unknown'
    category_value = str(row['CATEGORY']).strip()
    urgency_value = str(row['URGENCYCODE']).strip()

    return f"""Classify this IT ticket using EXACTLY this format:

TICKET:
Subject: {subject}
Problem: {problem}

CLASSIFICATION:
```json
{{
  "ISSUE": "{issue_value}",
  "CATEGORY": "{category_value}",
  "URGENCYCODE": "{urgency_value}"
}}
```"""

# --- Complete version of modified training script ---
import pandas as pd
import torch
import json
import gc
import os
import time
import psutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
import logging
from google.colab import drive
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Mount Google Drive ---
drive.mount('/content/drive')
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Datel_Project"
MODELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "saved_models")

# --- DATA LOADING AND PREPARATION ---
def load_and_prepare_data(filepath="Copy of Tickets_cleaned.xlsx"):
    logging.info(f"🔄 Loading and preparing data from '{filepath}'...")
    df = pd.read_excel(filepath)
    df = df.drop(columns=['ACCOUNTID', 'ACCOUNT', 'TICKETID', 'ALTERNATEKEYSUFFIX'], errors='ignore')
    df['SUBJECT'] = df['SUBJECT'].fillna('').astype(str)
    df['PROBLEM'] = df['PROBLEM'].fillna('').astype(str)

    # --- deal with missing value ---
    df['ISSUE'] = df['ISSUE'].fillna('Unknown').astype(str)
    df['CATEGORY'] = df['CATEGORY'].fillna('Unknown').astype(str)
    df['URGENCYCODE'] = df['URGENCYCODE'].fillna('3').astype(str)  # set it as medium level

    df['fullText'] = (df['SUBJECT'].str.strip() + " | " + df['PROBLEM'].str.strip()).str.strip()
    df = df[df['fullText'].str.len() > 10]

    # Print data analysis to understand categories
    print("=== DATA ANALYSIS ===")
    print("\nCATEGORY distribution:")
    print(df['CATEGORY'].value_counts())
    print("\nURGENCYCODE distribution:")
    print(df['URGENCYCODE'].value_counts())
    print("\nISSUE distribution (Top 10):")
    print(df['ISSUE'].value_counts().head(10))

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['URGENCYCODE'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['URGENCYCODE'])
    return train_df, val_df, test_df

# --- Simplified Prompt Engineering ---
def create_super_simple_prompt(row):
    subject = str(row['SUBJECT']).strip()
    problem = str(row['PROBLEM']).strip()

    issue_value = str(row['ISSUE']).strip() if str(row['ISSUE']).strip() != 'nan' else 'Unknown'
    category_value = str(row['CATEGORY']).strip()
    urgency_value = str(row['URGENCYCODE']).strip()

    return f"""Classify this ticket:

Subject: {subject}
Problem: {problem}

Output this exact format:
```json
{{
  "ISSUE": "{issue_value}",
  "CATEGORY": "{category_value}",
  "URGENCYCODE": "{urgency_value}"
}}
```"""

# --- MODEL TRAINING ---
def train_optimized_model(model_name, train_df, val_df=None):
    logging.info(f"🚀 Starting training for: {model_name}")
    start_time = time.time()

    train_df_copy = train_df.copy()
    # Use the simplified prompt
    train_df_copy['text'] = train_df_copy.apply(create_super_simple_prompt, axis=1)
    train_dataset = Dataset.from_pandas(train_df_copy[['text']])

    # Print some examples to see the format
    print("=== Training Examples ===")
    for i in range(min(3, len(train_dataset))):
        print(f"Example {i+1}:")
        print(train_dataset[i]['text'])
        print("-" * 50)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    output_dir = os.path.join(MODELS_DIR, f"{model_name.split('/')[-1]}_optimized_v15")  # v15 version
    training_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir=True, per_device_train_batch_size=1, gradient_accumulation_steps=16, gradient_checkpointing=True, learning_rate=5e-5, warmup_steps=100, weight_decay=0.01, num_train_epochs=3, save_strategy="epoch", save_total_limit=1, logging_steps=50, report_to="none", fp16=False, bf16=True, seed=42)

    def formatting_func(example):
        return example['text']

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        processing_class=tokenizer,
        peft_config=lora_config,
        formatting_func=formatting_func,
    )

    train_result = trainer.train()
    training_time = time.time() - start_time

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    training_metrics = {
        "training_loss": train_result.training_loss,
        "training_time_hours": training_time/3600,
        "total_steps": train_result.global_step,
        "model_name": model_name,
        "dataset_size": len(train_dataset),
    }

    with open(os.path.join(output_dir, "training_metrics.json"), 'w') as f:
        json.dump(training_metrics, f, indent=2)

    return trainer.model, tokenizer, output_dir

# --- MODEL EVALUATION ---
def evaluate_model_comprehensive(model, tokenizer, test_df, model_name, model_path):
    logging.info(f"🔄 Evaluating model: {model_name}")
    model.eval()
    predictions, true_labels = [], []
    eval_df = test_df.sample(n=min(200, len(test_df)), random_state=42)  # reduce training samples

    for _, row in eval_df.iterrows():
        true_labels.append({"ISSUE": row['ISSUE'], "CATEGORY": row['CATEGORY'], "URGENCYCODE": row['URGENCYCODE']})

        subject = str(row['SUBJECT']).strip()
        problem = str(row['PROBLEM']).strip()

        # Use the format same as training
        prompt = f"""Classify this ticket:

Subject: {subject}
Problem: {problem}

Output this exact format:
```json
"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            json_str = prediction_text.split("```json")[1].split("```")[0].strip()
            pred_labels = json.loads(json_str)
            predictions.append({
                "ISSUE": pred_labels.get("ISSUE", "Unknown"),
                "CATEGORY": pred_labels.get("CATEGORY", "Unknown"),
                "URGENCYCODE": str(pred_labels.get("URGENCYCODE", "Unknown"))
            })
        except (IndexError, json.JSONDecodeError):
            predictions.append({"ISSUE": "Parse Error", "CATEGORY": "Parse Error", "URGENCYCODE": "Parse Error"})

    # evaluation
    fields = ["ISSUE", "CATEGORY", "URGENCYCODE"]
    results = {}
    print(f"\n📊 {model_name} Evaluate result")

    for field in fields:
        y_true = [label[field] for label in true_labels]
        y_pred = [pred[field] for pred in predictions]
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        results[field] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }
        print(f"\n{field}:\n  Accuracy: {accuracy:.4f}\n  F1: {f1:.4f}\n  precision: {precision:.4f}\n  recall: {recall:.4f}")

    # overall result
    overall_accuracy = np.mean([results[field]["accuracy"] for field in fields])
    overall_f1 = np.mean([results[field]["f1_score"] for field in fields])
    overall_precision = np.mean([results[field]["precision"] for field in fields])
    overall_recall = np.mean([results[field]["recall"] for field in fields])

    results["overall"] = {
        "accuracy": overall_accuracy,
        "f1_score": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall
    }

    print(f"\nOverall evaluation:\n  Accuracy: {overall_accuracy:.4f}\n  F1: {overall_f1:.4f}\n  precision: {overall_precision:.4f}\n  recall: {overall_recall:.4f}")

    with open(os.path.join(model_path, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    train_data, val_data, test_data = load_and_prepare_data()
    if train_data is not None:
        models_to_run = ["google/gemma-2b-it"]  # run gemma first
        for model_name in models_to_run:
            try:
                trained_model, trained_tokenizer, model_path = train_optimized_model(model_name, train_data, val_data)
                evaluate_model_comprehensive(trained_model, trained_tokenizer, test_data, model_name, model_path)
                del trained_model, trained_tokenizer
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                logging.error(f"❌ Error processing {model_name}: {str(e)}", exc_info=True)
