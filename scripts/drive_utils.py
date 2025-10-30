# scripts/drive_utils.py
import io
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from google.oauth2 import service_account

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def build_drive_service(credentials_json_path):
    """
    Build a Google Drive API service using service account credentials.
    """
    try:
        if not os.path.exists(credentials_json_path):
            raise FileNotFoundError(f"Credentials file not found: {credentials_json_path}")

        print(f"[INFO] Initializing Google Drive service using: {credentials_json_path}")
        creds = service_account.Credentials.from_service_account_file(
            credentials_json_path, scopes=SCOPES
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        print("[INFO] Google Drive service initialized successfully.")
        return service

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
    except HttpError as e:
        print(f"[ERROR] Google API HTTP error: {e}")
    except Exception as e:
        print(f"[CRITICAL] Failed to initialize Google Drive service: {e}")
    return None


def list_files_in_folder(service, folder_id, mime_type=None):
    """
    List all files in a specific Google Drive folder.
    """
    if not service:
        print("[ERROR] No valid Google Drive service instance provided.")
        return []

    try:
        q = f"'{folder_id}' in parents and trashed=false"
        if mime_type:
            q += f" and mimeType='{mime_type}'"

        print(f"[INFO] Fetching files from folder ID: {folder_id}")
        results = service.files().list(
            q=q,
            pageSize=1000,
            fields="files(id,name,mimeType,modifiedTime)"
        ).execute()

        files = results.get("files", [])
        print(f"[INFO] Found {len(files)} file(s) in folder.")
        return files

    except HttpError as e:
        print(f"[ERROR] Google Drive API error while listing files: {e}")
    except Exception as e:
        print(f"[CRITICAL] Unexpected error while listing files: {e}")
    return []


def download_file(service, file_id, dest_path):
    """
    Download a single file from Google Drive to a specified local path.
    """
    if not service:
        print("[ERROR] No valid Google Drive service instance provided.")
        return None

    try:
        print(f"[INFO] Starting download for file ID: {file_id}")
        request = service.files().get_media(fileId=file_id)

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with io.FileIO(dest_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"[INFO] Download progress: {int(status.progress() * 100)}%")

        print(f"[INFO] File downloaded successfully to: {dest_path}")
        return dest_path

    except HttpError as e:
        print(f"[ERROR] Google Drive API error while downloading file {file_id}: {e}")
    except Exception as e:
        print(f"[CRITICAL] Failed to download file {file_id}: {e}")
    return None
