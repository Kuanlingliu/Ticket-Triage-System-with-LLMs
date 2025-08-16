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

def create_enhanced_vector_database(main_df, history_df, model):
    """
    Encodes the full conversation history of tickets into vectors 
    and saves them in a FAISS index for enhanced retrieval.
    """
    logging.info("Creating ENHANCED vector database for solution retrieval...")

    # --- Step 1: Process and aggregate the history data ---
    logging.info("Aggregating ticket conversation histories...")
    # Drop rows with no description
    history_df = history_df.dropna(subset=['ACTIVITYDESC']).copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # --- HIGHLIGHTED CHANGE: Convert all activity descriptions to string type ---
    # This line fixes the TypeError by ensuring all items are strings before joining.
    history_df['ACTIVITYDESC'] = history_df['ACTIVITYDESC'].astype(str)
    
    history_df['CREATEDATE'] = pd.to_datetime(history_df['CREATEDATE'])
    history_df = history_df.sort_values(by=['TICKETID', 'CREATEDATE'])
    
    # Group by TICKETID and join all descriptions into a single text block
    # This creates a full story for each ticket
    aggregated_history = history_df.groupby('TICKETID')['ACTIVITYDESC'].apply(
        lambda activities: "\n---\n".join(activities)
    ).reset_index()
    aggregated_history.rename(columns={'ACTIVITYDESC': 'FULL_HISTORY'}, inplace=True)

    # --- Step 2: Merge aggregated history back into the main dataframe ---
    logging.info("Merging full histories with main ticket data...")
    merged_df = pd.merge(main_df, aggregated_history, on='TICKETID', how='left')
    
    # --- Step 3: Create a single "knowledge" column ---
    # We prioritize the full history. If it doesn't exist, we fall back to the final SOLUTION.
    merged_df['KNOWLEDGE_TEXT'] = merged_df['FULL_HISTORY'].fillna(merged_df['SOLUTION'])
    
    # Filter out tickets that have no knowledge text at all
    df_knowledge = merged_df.dropna(subset=['KNOWLEDGE_TEXT']).copy()
    df_knowledge = df_knowledge[df_knowledge['KNOWLEDGE_TEXT'].str.strip() != '']

    if df_knowledge.empty:
        logging.warning("No knowledge text found to build the vector database. Skipping.")
        return

    # --- Step 4: Encode the new knowledge text and build the FAISS index ---
    logging.info(f"Encoding {len(df_knowledge)} enriched ticket knowledge documents...")
    knowledge_embeddings = model.encode(df_knowledge['KNOWLEDGE_TEXT'].tolist(), convert_to_tensor=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(knowledge_embeddings.shape[1])
    index.add(knowledge_embeddings.cpu().numpy())

    # --- Step 5: Save the new assets ---
    os.makedirs(ASSETS_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(ASSETS_DIR, "ticket_knowledge.index"))
    df_knowledge.to_pickle(os.path.join(ASSETS_DIR, "ticket_knowledge_data.pkl"))

    logging.info(f"✅ Enhanced vector database saved to '{ASSETS_DIR}'.")


if __name__ == "__main__":
    logging.info("--- Starting Enhanced Asset Building Process ---")

    # Define file paths
    main_tickets_file = "/content/drive/MyDrive/Datel_Project/Copy of Tickets_cleaned.xlsx"
    history_file = "/content/drive/MyDrive/Datel_Project/Ticket history 3.xlsx" # The new history file

    try:
        main_df = pd.read_excel(main_tickets_file)
        history_df = pd.read_excel(history_file)
    except FileNotFoundError as e:
        logging.error(f"FATAL: Could not find a data file. Error: {e}")
        exit()

    # Load the sentence transformer model for creating embeddings
    retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create the enhanced vector database using both dataframes
    create_enhanced_vector_database(main_df, history_df, retrieval_model)
    
    logging.info("--- Asset Building Complete ---")
    print("--- Asset Building Complete ---")

