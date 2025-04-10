import os
import sys
import re
import fnmatch
from google.cloud import aiplatform
from google.cloud import storage
from vertexai.preview import rag
from google.api_core.exceptions import NotFound
import logging
from dotenv import load_dotenv
from utils import load_raw_ignore_patterns
from pathlib import Path

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = "us-central1"
DEFAULT_GCS_BUCKET_NAME = f"{PROJECT_ID}-rag-code-store" if PROJECT_ID else "default-rag-code-store"
RAG_CORPUS_DISPLAY_NAME = "my-coding-agent-corpus"


def _validate_bucket_name_component(name_part: str) -> bool:
    if not name_part:
        return False
    return bool(re.match(r"^[a-z0-9-]+$", name_part))


def _ensure_gcs_bucket_exists(bucket_name: str, project_id: str, location: str):
    storage_client = storage.Client(project=project_id)
    try:
        bucket = storage_client.get_bucket(bucket_name)
        logging.info(f"Bucket {bucket_name} already exists.")
    except NotFound:
        logging.info(f"Bucket {bucket_name} not found. Creating...")
        try:
            bucket = storage_client.create_bucket(
                bucket_name,
                project=project_id,
                location=location,
            )
            bucket.iam_configuration.uniform_bucket_level_access_enabled = True
            bucket.patch()
            logging.info(
                f"Bucket {bucket_name} created in {location} with uniform access."
            )
        except Exception as e:
            logging.error(f"Failed to create bucket {bucket_name}: {e}")
            raise
    except Exception as e:
        logging.error(f"Error checking bucket {bucket_name}: {e}")
        raise


def upload_directory_to_gcs(local_directory_path: str,
                            bucket_name: str,
                            gcs_prefix: str = "code_upload/") -> str:
    if not os.path.isdir(local_directory_path):
        raise FileNotFoundError(
            f"Local directory not found: {local_directory_path}")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    local_directory_path = os.path.abspath(local_directory_path)
    ignore_file = os.path.join(local_directory_path, ".indexignore")
    ignore_patterns = load_raw_ignore_patterns(ignore_file)
    standard_ignores = [
        '.git/', '.venv/', 'venv/', '__pycache__/', 'node_modules/', '.DS_Store'
    ]
    ignore_patterns.extend(standard_ignores)
    logging.info(
        f"Uploading files from '{local_directory_path}' to gs://{bucket_name}/{gcs_prefix}..."
    )
    uploaded_files = 0
    skipped_files = 0
    ignored_files = 0
    ignored_dirs_count = 0
    for root, dirs, files in os.walk(local_directory_path, topdown=True):
        original_dirs = list(dirs)
        dirs[:] = []
        for d in original_dirs:
            rel_dir_path = os.path.relpath(os.path.join(root, d),
                                           local_directory_path)
            rel_dir_path_normalized = str(Path(rel_dir_path).as_posix())
            is_ignored = any(
                fnmatch.fnmatch(rel_dir_path_normalized +
                                '/', pattern) or fnmatch.fnmatch(d, pattern)
                for pattern in ignore_patterns)
            if is_ignored:
                logging.info(
                    f"  Ignoring directory: {rel_dir_path_normalized}/")
                ignored_dirs_count += 1
            else:
                dirs.append(d)
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path_for_match = os.path.relpath(local_path,
                                                      local_directory_path)
            relative_path_normalized = str(
                Path(relative_path_for_match).as_posix())
            is_ignored = any(
                fnmatch.fnmatch(relative_path_normalized, pattern) or
                fnmatch.fnmatch(filename, pattern)
                for pattern in ignore_patterns)
            if is_ignored:
                logging.info(f"  Ignoring file: {relative_path_normalized}")
                ignored_files += 1
                continue
            gcs_path = os.path.join(gcs_prefix,
                                    relative_path_normalized).replace(
                                        "\\\\", "/")
            try:
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                uploaded_files += 1
                if uploaded_files % 100 == 0:
                    logging.info(f"  Uploaded {uploaded_files} files...")
            except Exception as e:
                logging.warning(
                    f"  Failed to upload {local_path} to {gcs_path}: {e}")
                skipped_files += 1
    logging.info(
        f"Finished uploading. Uploaded: {uploaded_files}, Skipped (errors): {skipped_files}, Ignored Dirs: {ignored_dirs_count}, Ignored Files: {ignored_files}."
    )
    if skipped_files > 0:
        logging.warning(f"{skipped_files} files were skipped due to errors.")
    return f"gs://{bucket_name}/{gcs_prefix}"


def setup_rag_for_directory(local_directory_path: str,
                            gcs_bucket_name: str) -> str | None:
    if not PROJECT_ID:
        logging.error("GOOGLE_CLOUD_PROJECT environment variable not set.")
        raise ValueError("Set the GOOGLE_CLOUD_PROJECT environment variable.")
    if not gcs_bucket_name:
        raise ValueError("GCS bucket name must be provided.")
    if not os.path.isdir(local_directory_path):
        logging.error(
            f"Error: Provided path is not a valid directory: {local_directory_path}"
        )
        return None
    local_directory_path = os.path.abspath(local_directory_path)
    logging.info(f"Setting up RAG for directory: {local_directory_path}")
    logging.info(
        f"Using Project: {PROJECT_ID}, Location: {LOCATION}, Bucket: {gcs_bucket_name}"
    )
    try:
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI SDK: {e}")
        return None
    try:
        _ensure_gcs_bucket_exists(gcs_bucket_name, PROJECT_ID, LOCATION)
    except Exception as e:
        logging.error(f"Failed proceeding due to GCS bucket setup error: {e}")
        return None
    gcs_uri_prefix = ""
    try:
        gcs_uri_prefix = upload_directory_to_gcs(local_directory_path,
                                                 gcs_bucket_name,
                                                 gcs_prefix="code_upload/")
    except FileNotFoundError as e:
        logging.error(e)
        return None
    except Exception as e:
        logging.error(f"Error uploading to GCS: {e}")
        return None
    rag_corpus = None
    corpus_id = None
    try:
        corpora = rag.list_corpora()
        for c in corpora:
            if c.display_name == RAG_CORPUS_DISPLAY_NAME:
                rag_corpus = c
                corpus_id = c.name.split('/')[-1]
                logging.info(
                    f"Found existing RAG Corpus: {rag_corpus.name} (ID: {corpus_id})"
                )
                break
        if not rag_corpus:
            logging.info(f"Creating RAG Corpus: {RAG_CORPUS_DISPLAY_NAME}")
            rag_corpus = rag.create_corpus(
                display_name=RAG_CORPUS_DISPLAY_NAME,
                description="Corpus containing code files for AI coding agent.")
            corpus_id = rag_corpus.name.split('/')[-1]
            logging.info(
                f"Created RAG Corpus: {rag_corpus.name} (ID: {corpus_id})")
    except Exception as e:
        logging.error(f"Error finding or creating RAG Corpus: {e}")
        return None
    try:
        logging.info(
            f"Starting RAG file import from {gcs_uri_prefix} into {rag_corpus.name}..."
        )
        import_op = rag.import_files(
            corpus_name=rag_corpus.name,
            paths=[gcs_uri_prefix],
            chunk_size=512,
            chunk_overlap=100,
        )
        logging.info(
            f"Import operation initiated. Result (if any): {import_op}")
        logging.info(
            f"File import and indexing likely completed or running for Corpus ID: {corpus_id}."
        )
        return rag_corpus.name
    except Exception as e:
        logging.error(
            f"Error importing files into RAG Corpus {rag_corpus.name}: {e}")
        return None


if __name__ == "__main__":
    if not PROJECT_ID:
        print("Error: GOOGLE_CLOUD_PROJECT environment variable is not set.")
        sys.exit(1)
    user_identifier = ""
    while not user_identifier:
        user_input = input(
            "Enter a unique identifier for your project (e.g., 'my-agent-proj', used for GCS bucket name): "
        ).strip()
        user_identifier = user_input.lower().replace(" ", "-")
        if not _validate_bucket_name_component(user_identifier):
            print(
                "Invalid identifier. Use only lowercase letters, numbers, and hyphens."
            )
            user_identifier = ""
        elif len(user_identifier) < 3 or len(user_identifier) > 20:
            print("Identifier length must be between 3 and 20 characters.")
            user_identifier = ""
    final_bucket_name = f"{user_identifier}-rag-code-store"
    print(f"Will use GCS Bucket: gs://{final_bucket_name}")
    if len(sys.argv) < 2:
        print("\nUsage: python rag_setup.py <path_to_local_code_directory>")
        test_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..'))
        print(f"No directory provided. TESTING with: {test_dir}")
        if not os.path.isdir(test_dir):
            print(f"Test directory {test_dir} does not exist. Exiting.")
            sys.exit(1)
        local_dir = test_dir
    else:
        local_dir = sys.argv[1]
    print("\n--- Starting RAG Setup ---")
    corpus_name = setup_rag_for_directory(local_dir, final_bucket_name)
    print("\n--- RAG Setup Finished ---")
    if corpus_name:
        print(f"Successfully set up RAG. Corpus Name: {corpus_name}")
    else:
        print("RAG setup failed. Check logs for details.")
        sys.exit(1)
