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
MODELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "saved_models", "bert_model_v2") # New version folder
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

    # --- HIGHLIGHTED CHANGE: Create a 3-way split (train, validation, test) ---
    # First, split into training (70%) and a temporary set (30%)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    # Then, split the temporary set into validation (15%) and test (15%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    logging.info(f"✅ Data loaded. Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)} rows.")
    return train_df, val_df, test_df

# --- BERT MODEL TRAINING & EVALUATION ---
def train_bert_classifier(train_df, val_df, test_df, model_name="bert-base-uncased"):
    """
    Fine-tunes and evaluates a BERT model using separate validation and test sets.
    """
    logging.info(f"🚀 Starting BERT training for: {model_name}")

    # --- 1. Encode Labels ---
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_df['labels'])
    val_labels = mlb.transform(val_df['labels'])
    test_labels = mlb.transform(test_df['labels'])

    # --- 2. Save the Binarizer for Reuse ---
    output_dir = os.path.join(MODELS_DIR, f"{model_name.split('/')[-1]}_triage_v16")
    os.makedirs(output_dir, exist_ok=True)
    binarizer_path = os.path.join(output_dir, "mlb_binarizer.joblib")
    joblib.dump(mlb, binarizer_path)
    logging.info(f"💾 Label binarizer saved to {binarizer_path}")

    # --- 3. Tokenize Text ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_df['fullText'].tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_df['fullText'].tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_df['fullText'].tolist(), truncation=True, padding=True, max_length=512)

    # --- 4. Create PyTorch Datasets ---
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

    # --- 5. Define Evaluation Metrics ---
    def compute_metrics(p: EvalPrediction):
        logits = p.predictions
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs > 0.5)] = 1

        f1_micro = f1_score(y_true=p.label_ids, y_pred=predictions, average="micro")
        precision_micro = precision_score(y_true=p.label_ids, y_pred=predictions, average="micro")
        recall_micro = recall_score(y_true=p.label_ids, y_pred=predictions, average="micro")

        return {
            "f1_micro": f1_micro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
        }

    # --- 6. Configure and Train the Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, # This will use the validation set to find the best model
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, # --- HIGHLIGHTED CHANGE: Use validation set for in-training evaluation ---
        compute_metrics=compute_metrics,
    )

    logging.info("🏋️‍♀️ Fine-tuning BERT model...")
    trainer.train()

    # --- 7. Final Evaluation and Saving ---
    logging.info("📊 Performing final, unbiased evaluation on the separate test set...")
    eval_results = trainer.evaluate(eval_dataset=test_dataset) # --- HIGHLIGHTED CHANGE: Final evaluation on test set ---

    print("\n" + "="*50)
    print("📊 FINAL BERT EVALUATION RESULTS (ON TEST SET)")
    print("="*50)
    for key, value in eval_results.items():
        print(f"  - {key}: {value:.4f}")
    print("="*50)

    # Save the final results to a JSON file
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    logging.info(f"💾 Evaluation results saved to {results_path}")

    # Save the fine-tuned model and tokenizer
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

    # --- HIGHLIGHTED CHANGE: Load all three datasets ---
    train_data, val_data, test_data = load_and_prepare_data()

    if train_data is not None:
        # --- HIGHLIGHTED CHANGE: Pass all three datasets to the function ---
        train_bert_classifier(train_data, val_data, test_data)
        logging.info("🎉 BERT pipeline completed successfully.")

