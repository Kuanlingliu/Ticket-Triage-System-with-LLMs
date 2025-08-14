## new version
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import logging
import os
from google.colab import drive

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Mount Google Drive ---
drive.mount('/content/drive')
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Datel_Project"
ASSETS_DIR = os.path.join(DRIVE_PROJECT_PATH, "assets")

def create_vector_database(df, model):
    """Encodes solutions into vectors and saves them in a FAISS index for fast retrieval."""
    logging.info("Creating vector database for solution retrieval...")

    if 'SOLUTION' not in df.columns:
        logging.warning("'SOLUTION' column not found. Skipping vector database creation.")
        return

    df_solved = df.dropna(subset=['SOLUTION']).copy()
    df_solved = df_solved[df_solved['SOLUTION'].str.strip() != '']

    if df_solved.empty:
        logging.warning("No solved tickets found to build the vector database. Skipping.")
        return

    logging.info(f"Encoding {len(df_solved)} ticket solutions...")
    solution_embeddings = model.encode(df_solved['SOLUTION'].tolist(), convert_to_tensor=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(solution_embeddings.shape[1])
    index.add(solution_embeddings.cpu().numpy())

    os.makedirs(ASSETS_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(ASSETS_DIR, "ticket_solutions.index"))
    df_solved.to_pickle(os.path.join(ASSETS_DIR, "ticket_data.pkl"))

    logging.info(f"✅ Vector database saved to '{ASSETS_DIR}'.")

# --- HIGHLIGHTED CHANGE: Removed the create_competency_matrix function ---
# As per the action plan, this feature is for future consideration and not needed for the current prototype.

if __name__ == "__main__":
    logging.info("--- Starting Asset Building Process ---")

    data_file = "/content/drive/MyDrive/Datel_Project/Copy of Tickets_cleaned.xlsx"
    try:
        main_df = pd.read_excel(data_file)
    except FileNotFoundError:
        logging.error(f"FATAL: '{data_file}' not found. Please ensure the file exists at this path.")
        exit()

    retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create the vector database, which is essential for the RAG system
    create_vector_database(main_df, retrieval_model)

    # --- HIGHLIGHTED CHANGE: Removed the call to create_competency_matrix ---

    logging.info("--- Asset Building Complete ---")
    print("--- Asset Building Complete ---")
