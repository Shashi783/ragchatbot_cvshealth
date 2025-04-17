import logging
import boto3
from pathlib import Path
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError
import shutil
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages S3 storage operations including persist directory and chat history."""

    _s3_client = None
    _chat_hist = None
    _chat_id = None
    _persist_dir_cache = {}

    def __init__(self, config: Dict[str, Any], secret: Dict[str, Any]):
        self._config = config
        self._secret = secret
        self._init_s3_client()

    def _init_s3_client(self) -> boto3.client:
        """Initialize S3 client with credentials."""
        if StorageManager._s3_client is None:
            try:
                logger.info("Initializing S3 client...")
                StorageManager._s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self._secret["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=self._secret["AWS_SECRET_ACCESS_KEY"],
                    region_name=self._config["AWS_REGION"],
                )
            except ClientError as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error initializing S3 client: {e}")
                raise

    def load_persist_dir(self, persist_dir: str, collection_name: str) -> str:
        """Load persist directory from S3."""
        cache_key = f"{persist_dir}/{collection_name}"
        if cache_key in StorageManager._persist_dir_cache:
            logger.info(f"Using cached persist directory: {StorageManager._persist_dir_cache[cache_key]}")
            return StorageManager._persist_dir_cache[cache_key]

        try:
            logger.info(f"Loading persist directory for collection: {collection_name}")
            s3_persist_path = f"{persist_dir}/{collection_name}"

            s3_response = StorageManager._s3_client.list_objects_v2(
                Bucket=self._config["S3_PERSIST_BUCKET"], Prefix=s3_persist_path
            )

            if "Contents" not in s3_response:
                logger.error(f"Persist directory not found: {s3_persist_path}")
                raise ValueError(f"Persist directory {s3_persist_path} not found")

            local_persist_dir = Path(self._config["S3_PERSIST_DIR"], persist_dir, collection_name)
            local_persist_dir.mkdir(parents=True, exist_ok=True)

            for obj in s3_response["Contents"]:
                if obj["Key"].endswith("/"):
                    continue

                file_path = Path(self._config["S3_PERSIST_DIR"], obj["Key"])
                if file_path.exists():
                    logger.info(f"Deleting existing file: {file_path}")
                    file_path.unlink()

                logger.info(f"Downloading file: {file_path}")
                file_path.parent.mkdir(parents=True, exist_ok=True)

                StorageManager._s3_client.download_file(
                    self._config["S3_PERSIST_BUCKET"], obj["Key"], str(file_path)
                )

            logger.info(f"Successfully loaded persist directory to {local_persist_dir}")
            StorageManager._persist_dir_cache[cache_key] = local_persist_dir
            return local_persist_dir

        except ClientError as e:
            logger.error(f"S3 error loading persist directory: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading persist directory: {e}")
            raise

    def load_chat_history(self, curr_chat_id: str) -> Optional[str]:
        """Load chat history for given chat ID."""
        try:
            logger.info(f"Loading chat history for chat ID: {curr_chat_id}")

            if curr_chat_id == StorageManager._chat_id and StorageManager._chat_hist is not None:
                logger.debug("Returning cached chat history")
                return StorageManager._chat_hist

            StorageManager._chat_id = curr_chat_id
            chat_summ_file = f"{self._config['S3_CHAT_HISTORY']}/{StorageManager._chat_id}.md"

            s3_chat_summ_obj = StorageManager._s3_client.list_objects_v2(
                Bucket=self._config["S3_LOGS_BUCKET"],
                Delimiter="/",
                Prefix=f"{self._config['S3_CHAT_HISTORY']}/",
            )

            if "Contents" not in s3_chat_summ_obj:
                logger.info("Creating new chat history directory")
                StorageManager._s3_client.put_object(
                    Bucket=self._config["S3_LOGS_BUCKET"],
                    Key=f"{self._config['S3_CHAT_HISTORY']}/",
                )
            elif chat_summ_file in [content["Key"] for content in s3_chat_summ_obj["Contents"]]:
                logger.info("Loading existing chat history")
                response = StorageManager._s3_client.get_object(
                    Bucket=self._config["S3_LOGS_BUCKET"], Key=chat_summ_file
                )
                StorageManager._chat_hist = response["Body"].read().decode("utf-8")

        except ClientError as e:
            logger.error(f"S3 error loading chat history: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading chat history: {str(e)}")
            raise

    @property
    def chat_hist(self) -> Optional[str]:
        """Get current chat history."""
        return self._chat_hist

    @chat_hist.setter
    def chat_hist(self, value: str):
        """Set chat history."""
        self._chat_hist = value 

class LocalStorageManager:
    """Manages local storage operations for persist directory and chat history."""

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._chat_hist = None
        self._chat_id = None
        self._persist_dir_cache = {}
        
        # Ensure base directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        try:
            # Ensure temp directory exists
            temp_dir = self._config.get("LOCAL_TEMP_DIR", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            logger.info(f"Ensured temp directory exists: {temp_dir}")
            
            # Ensure storage directory exists
            storage_dir = self._config.get("LOCAL_STORAGE_DIR", "storage")
            os.makedirs(storage_dir, exist_ok=True)
            logger.info(f"Ensured storage directory exists: {storage_dir}")
            
            # Ensure persist directory exists
            persist_dir = self._config.get("LOCAL_PERSIST_DIR", "persist")
            os.makedirs(persist_dir, exist_ok=True)
            logger.info(f"Ensured persist directory exists: {persist_dir}")
            
            # Ensure chat history directory exists
            chat_history_dir = self._config.get("LOCAL_CHAT_HISTORY_DIR", "chat_history")
            os.makedirs(chat_history_dir, exist_ok=True)
            logger.info(f"Ensured chat history directory exists: {chat_history_dir}")
            
        except Exception as e:
            logger.error(f"Error ensuring directories: {str(e)}")
            raise

    def load_persist_dir(self, persist_dir: str, collection_name: str) -> str:
        """Load persist directory for a collection."""
        cache_key = f"{persist_dir}/{collection_name}"
        if cache_key in self._persist_dir_cache:
            logger.info(f"Using cached persist directory: {self._persist_dir_cache[cache_key]}")
            return self._persist_dir_cache[cache_key]
        try:
            logger.info(f"Loading persist directory for collection: {collection_name}")
            source_path = Path(self._config["LOCAL_PERSIST_DIR"], collection_name)
            target_path = Path(self._config["LOCAL_TEMP_DIR"], collection_name)
            # print(source_path)
            # print(os.getcwd())
            # print(os.listdir(os.path.dirname(__file__)))

            if not source_path.exists():
                logger.error(f"Persist directory not found: {source_path}")
                raise ValueError(f"Persist directory {source_path} not found")

            # Clean up existing target directory if it exists
            if target_path.exists():
                logger.info(f"Cleaning up existing target directory: {target_path}")
                shutil.rmtree(target_path)

            # Copy persist directory to temp location
            logger.info(f"Copying persist directory from {source_path} to {target_path}")
            shutil.copytree(source_path, target_path)

            # Verify essential files exist
            required_files = ["docstore.json", "graph_store.json", "index_store.json", "vector_store.json"]
            for file in required_files:
                if not (target_path / file).exists():
                    logger.warning(f"Required file {file} not found in persist directory")

            logger.info(f"Successfully loaded persist directory to {target_path}")
            self._persist_dir_cache[cache_key] = target_path
            return target_path

        except Exception as e:
            logger.error(f"Error loading persist directory: {e}")
            raise

    def load_chat_history(self, curr_chat_id: str) -> Optional[str]:
        """Load chat history for given chat ID."""
        try:
            logger.info(f"Loading chat history for chat ID: {curr_chat_id}")

            if curr_chat_id == self._chat_id and self._chat_hist is not None:
                logger.debug("Returning cached chat history")
                return self._chat_hist

            self._chat_id = curr_chat_id
            chat_file = Path(self._config["LOCAL_CHAT_HISTORY_DIR"], f"{self._chat_id}.json")

            if chat_file.exists():
                logger.info("Loading existing chat history")
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                    self._chat_hist = chat_data.get('history', '')
            else:
                logger.info("Creating new chat history")
                self._chat_hist = None

            return self._chat_hist

        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            raise

    @property
    def chat_hist(self) -> Optional[str]:
        """Get current chat history."""
        return self._chat_hist

    @chat_hist.setter
    def chat_hist(self, value: str):
        """Set chat history and save to file."""
        self._chat_hist = value
        if self._chat_id:
            chat_file = Path(self._config["LOCAL_CHAT_HISTORY_DIR"], f"{self._chat_id}.json")
            chat_data = {
                'history': value,
                'last_updated': str(datetime.utcnow())
            }
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, indent=2)
            logger.info(f"Saved chat history to {chat_file}")

    def cleanup_temp_files(self) -> None:
        """Clean up temporary files and directories."""
        try:
            temp_dir = Path(self._config["LOCAL_TEMP_DIR"])
            if temp_dir.exists():
                logger.info(f"Cleaning up temp directory: {temp_dir}")
                shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Successfully cleaned up temp directory")

        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
            raise

    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.cleanup_temp_files() 