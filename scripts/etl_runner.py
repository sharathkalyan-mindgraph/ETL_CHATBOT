# scripts/etl_runner.py
import os
import json
from dotenv import load_dotenv

load_dotenv()

from scripts.drive_utils import build_drive_service, list_files_in_folder, download_file
from scripts.etl import doc_to_chunks
from scripts.embeddings import get_embeddings_provider
from scripts.vectorstore import create_chroma_from_chunks

INGESTED_MAP = "./data/ingested_files.json"


def load_ingested_map():
    """Load the map of ingested files from disk."""
    try:
        if os.path.exists(INGESTED_MAP):
            with open(INGESTED_MAP, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load ingested map: {e}")
    return {}


def save_ingested_map(m):
    """Save the updated ingested map."""
    try:
        os.makedirs(os.path.dirname(INGESTED_MAP), exist_ok=True)
        with open(INGESTED_MAP, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Could not save ingested map: {e}")


def run_etl(drive_folder_id, creds_json_path, chunk_size=800):
    """Main ETL runner to extract, transform, and load data from Google Drive."""
    try:
        print("[INFO] Building Google Drive service...")
        service = build_drive_service(creds_json_path)
    except Exception as e:
        print(f"[CRITICAL] Failed to build Google Drive service: {e}")
        return

    try:
        files = list_files_in_folder(service, drive_folder_id)
        if not files:
            print("[WARN] No files found in the specified Drive folder.")
            return
        print(f"[INFO] Found {len(files)} file(s) in Drive folder.")
    except Exception as e:
        print(f"[ERROR] Failed to list files in Drive folder: {e}")
        return

    try:
        embeddings = get_embeddings_provider()
        print("[INFO] Embedding provider initialized successfully.")
    except Exception as e:
        print(f"[CRITICAL] Failed to initialize embeddings provider: {e}")
        return

    ingested = load_ingested_map()
    all_chunks = []

    for f in files:
        try:
            fid = f.get("id")
            fname = f.get("name", "Unnamed")
            modified = f.get("modifiedTime")

            if not fid:
                print(f"[WARN] Skipping file with missing ID: {fname}")
                continue

            if fid in ingested and ingested[fid] == modified:
                print(f"[INFO] Skipping unchanged file: {fname}")
                continue

            print(f"[INFO] Downloading {fname} ({fid})...")
            local_path = os.path.join(os.getenv("TEMP", "C:\\Windows\\Temp"), f"{fid}_{fname}")

            try:
                download_file(service, fid, local_path)
                if not os.path.exists(local_path):
                    print(f"[ERROR] File download failed for: {fname}")
                    continue
            except Exception as e:
                print(f"[ERROR] Failed to download {fname}: {e}")
                continue

            # Extract and chunk
            try:
                chunks = doc_to_chunks(local_path, fid, chunk_size=chunk_size)
                if not chunks:
                    print(f"[WARN] No text extracted from: {fname}")
                    continue

                for c in chunks:
                    c["metadata"].update({
                        "file_name": fname,
                        "mimeType": f.get("mimeType"),
                        "modifiedTime": modified
                    })
                all_chunks.extend(chunks)
                ingested[fid] = modified
                print(f"[INFO] Processed {len(chunks)} chunks from {fname}")
            except Exception as e:
                print(f"[ERROR] Failed to process {fname}: {e}")
                continue

        except Exception as e:
            print(f"[CRITICAL] Unexpected error while handling file {f}: {e}")

    if all_chunks:
        try:
            vectordb = create_chroma_from_chunks(all_chunks, embeddings)
            if vectordb:
                print(f"[INFO] Ingested {len(all_chunks)} new chunks into Chroma at {os.getenv('CHROMA_DB_DIR')}")
            else:
                print("[ERROR] Failed to create or update Chroma vector store.")
        except Exception as e:
            print(f"[CRITICAL] Failed to update Chroma vector store: {e}")
    else:
        print("[INFO] No new chunks to ingest.")

    # Save ingestion state safely
    try:
        save_ingested_map(ingested)
    except Exception as e:
        print(f"[ERROR] Failed to save ingestion map after ETL: {e}")

    print("[INFO] ETL run completed.")


if __name__ == "__main__":
    try:
        DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "1B5lPTwQcRC-ZLEDtixOwg9_ZdA0-ADUU")
        CREDS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./service-account.json")
        print("[INFO] Starting ETL process...")
        run_etl(DRIVE_FOLDER_ID, CREDS_PATH)
        print("[INFO] ETL process complete.")
    except Exception as e:
        print(f"[FATAL] Unhandled exception in main execution: {e}")
