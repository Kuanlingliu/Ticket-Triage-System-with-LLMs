import pandas as pd
import torch
import json
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import Dataset
import logging
from google.colab import drive
import numpy as np
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Mount Google Drive & Define Paths ---
drive.mount('/content/drive')
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Datel_Project"
MODELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "saved_models", "bert_model_v3") # New version folder
os.makedirs(MODELS_DIR, exist_ok=True)

# --- DATA LOADING AND PREPARATION ---
def load_and_prepare_data(filepath="/content/drive/MyDrive/Datel_Project/Copy of Tickets_cleaned.xlsx"):
    """Loads and prepares data, splitting it into train, validation, and test sets."""
    logging.info(f"🔄 Loading and preparing data from '{filepath}'...")
    df = pd.read_excel(filepath)
    df = df.drop(columns=['ACCOUNTID', 'ACCOUNT', 'TICKETID', 'ALTERNATEKEYSUFFIX'], errors='ignore')
    
    # Clean and fill missing values
    df['SUBJECT'] = df['SUBJECT'].fillna('').astype(str)
    df['PROBLEM'] = df['PROBLEM'].fillna('').astype(str)
    df['ISSUE'] = df['ISSUE'].fillna('Unknown').astype(str)
    df['CATEGORY'] = df['CATEGORY'].fillna('Unknown').astype(str)
    df['URGENCYCODE'] = df['URGENCYCODE'].fillna('3').astype(str)

    df['fullText'] = (df['SUBJECT'].str.strip() + " | " + df['PROBLEM'].str.strip()).str.strip()
    df = df[df['fullText'].str.len() > 10]

    df['labels'] = df.apply(
        lambda row: [
            f"ISSUE:{row['ISSUE']}",
            f"CATEGORY:{row['CATEGORY']}",
            f"URGENCYCODE:{row['URGENCYCODE']}"
        ],
        axis=1
    )
    
    # Create a 3-way split (train, validation, test)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    logging.info(f"✅ Data loaded. Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)} rows.")
    return train_df, val_df, test_df

# --- HIGHLIGHTED CHANGE: New function for detailed, per-category evaluation ---
def evaluate_bert_macro(trainer, test_dataset, mlb, output_dir):
    """
    Performs a detailed evaluation by calculating metrics for each category separately.
    """
    logging.info("📊 Performing detailed, per-category (macro) evaluation...")

    # 1. Get model predictions
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    true_labels_binary = predictions_output.label_ids

    # 2. Convert logits to binary predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    predictions_binary = np.zeros(probs.shape)
    predictions_binary[np.where(probs > 0.5)] = 1

    # 3. Decode binary predictions back to text labels using the binarizer
    predicted_labels_text = mlb.inverse_transform(predictions_binary)
    true_labels_text = mlb.inverse_transform(true_labels_binary)

    # 4. Parse the text labels into dictionaries
    def parse_labels(label_tuples):
        parsed = []
        for label_tuple in label_tuples:
            d = {"ISSUE": "Unknown", "CATEGORY": "Unknown", "URGENCYCODE": "Unknown"}
            for label in label_tuple:
                key, value = label.split(":", 1)
                d[key] = value
            parsed.append(d)
        return parsed

    y_pred_parsed = parse_labels(predicted_labels_text)
    y_true_parsed = parse_labels(true_labels_text)

    # 5. Calculate metrics for each field
    fields = ["ISSUE", "CATEGORY", "URGENCYCODE"]
    results = {}
    
    print("\n" + "="*50)
    print("📊 DETAILED BERT EVALUATION (MACRO)")
    print("="*50)

    for field in fields:
        y_true = [d[field] for d in y_true_parsed]
        y_pred = [d[field] for d in y_pred_parsed]
        
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
        print(f"\n{field} Classification:")
        print(f"  - Accuracy:  {accuracy:.4f}")
        print(f"  - F1 Score:  {f1:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall:    {recall:.4f}")

    # 6. Calculate and print overall macro-average results
    overall_accuracy = np.mean([results[field]["accuracy"] for field in fields])
    overall_f1 = np.mean([results[field]["f1_score"] for field in fields])
    # --- ADDED PRECISION AND RECALL CALCULATION ---
    overall_precision = np.mean([results[field]["precision"] for field in fields])
    overall_recall = np.mean([results[field]["recall"] for field in fields])
    
    results["overall_macro_average"] = {
        "accuracy": overall_accuracy,
        "f1_score": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall
    }
    
    print("\n" + "-"*50)
    print("OVERALL MACRO-AVERAGE:")
    print(f"  - Avg Accuracy: {overall_accuracy:.4f}")
    print(f"  - Avg F1 Score: {overall_f1:.4f}")
    # --- ADDED PRECISION AND RECALL TO PRINTOUT ---
    print(f"  - Avg Precision: {overall_precision:.4f}")
    print(f"  - Avg Recall: {overall_recall:.4f}")
    print("="*50)

    # 7. Save the detailed results to a new file
    results_path = os.path.join(output_dir, "evaluation_results_detailed.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"💾 Detailed evaluation results saved to {results_path}")


# --- BERT MODEL TRAINING & EVALUATION ---
def train_bert_classifier(train_df, val_df, test_df, model_name="bert-base-uncased"):
    logging.info(f"🚀 Starting BERT training for: {model_name}")
    
    # Encode Labels
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_df['labels'])
    val_labels = mlb.transform(val_df['labels'])
    test_labels = mlb.transform(test_df['labels'])
    
    # Save the Binarizer
    output_dir = os.path.join(MODELS_DIR, f"{model_name.split('/')[-1]}_triage_v16")
    os.makedirs(output_dir, exist_ok=True)
    binarizer_path = os.path.join(output_dir, "mlb_binarizer.joblib")
    joblib.dump(mlb, binarizer_path)
    logging.info(f"💾 Label binarizer saved to {binarizer_path}")

    # Tokenize Text
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_df['fullText'].tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_df['fullText'].tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_df['fullText'].tolist(), truncation=True, padding=True, max_length=512)

    # Create PyTorch Datasets
    class TicketDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = TicketDataset(train_encodings, train_labels)
    val_dataset = TicketDataset(val_encodings, val_labels)
    test_dataset = TicketDataset(test_encodings, test_labels)

    # Define Micro-Average Metrics for Trainer
    def compute_metrics_micro(p: EvalPrediction):
        logits = p.predictions
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs > 0.5)] = 1
        f1_micro = f1_score(y_true=p.label_ids, y_pred=predictions, average="micro")
        return {"f1_micro": f1_micro}

    # Configure and Train the Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(mlb.classes_), problem_type="multi_label_classification")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_micro,
    )

    logging.info("🏋️‍♀️ Fine-tuning BERT model...")
    trainer.train()

    # --- Final Evaluation and Saving ---
    logging.info("📊 Performing final micro-average evaluation...")
    eval_results_micro = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\n" + "="*50)
    print("📊 FINAL BERT EVALUATION (MICRO-AVERAGE)")
    print("="*50)
    for key, value in eval_results_micro.items():
        print(f"  - {key}: {value:.4f}")
    print("="*50)

    results_path_micro = os.path.join(output_dir, "evaluation_results_micro.json")
    with open(results_path_micro, 'w') as f:
        json.dump(eval_results_micro, f, indent=2)
    logging.info(f"💾 Micro-average results saved to {results_path_micro}")

    # --- HIGHLIGHTED CHANGE: Call the new macro evaluation function ---
    evaluate_bert_macro(trainer, test_dataset, mlb, output_dir)

    # Save the final model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"✅ Model, tokenizer, and binarizer saved to {output_dir}")

# --- MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    logging.info("🚀 Starting BERT Training and Evaluation Pipeline")
    if not torch.cuda.is_available():
        logging.error("❌ GPU not available. This script requires a GPU.")
    else:
        logging.info(f"✅ GPU detected: {torch.cuda.get_device_name()}")

    train_data, val_data, test_data = load_and_prepare_data()
    if train_data is not None:
        train_bert_classifier(train_data, val_data, test_data)
        logging.info("🎉 BERT pipeline completed successfully.")
