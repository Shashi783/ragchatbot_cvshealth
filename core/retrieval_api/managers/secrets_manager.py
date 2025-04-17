import os
import json
import yaml
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_secret(secret_name: str) -> Optional[str]:
    """Get a secret value from environment variables or secrets file.
    
    Args:
        secret_name: Name of the secret to retrieve
        
    Returns:
        The secret value if found, None otherwise
    """
    try:
        # First try environment variables
        if secret_name in os.environ:
            return os.environ[secret_name]

        # Then try secrets file
        secrets_file = os.path.join(os.path.dirname(__file__), "..", "..", "secrets.json")
        if os.path.exists(secrets_file):
            with open(secrets_file, "r") as f:
                secrets = json.load(f)
                return secrets.get(secret_name)

        logger.warning(f"Secret {secret_name} not found in environment or secrets file")
        return None

    except Exception as e:
        logger.error(f"Error retrieving secret {secret_name}: {e}")
        return None

def get_config() -> Dict[str, Any]:
    """Get configuration from environment or config file.
    
    Returns:
        Configuration dictionary
    """
    try:
        # First try environment variables
        config = {}
        for key, value in os.environ.items():
            if key.startswith("APP_"):
                config[key[4:].lower()] = value

        # Then try config files
        # Get the path to the retrieval_api directory
        retrieval_api_dir = os.path.dirname(os.path.dirname(__file__))

        # Try config.yaml first
        yaml_config = os.path.join(retrieval_api_dir, "config", "config.yaml")
        if os.path.exists(yaml_config):
            with open(yaml_config, "r") as f:
                yaml_config_dict = yaml.safe_load(f)
                config.update(yaml_config_dict)
                logger.info(f"Loaded configuration from {yaml_config}")
                logger.debug(f"Configuration: {config}")

        # Then try config.json as fallback
        json_config = os.path.join(retrieval_api_dir, "config.json")
        if os.path.exists(json_config):
            with open(json_config, "r") as f:
                json_config_dict = json.load(f)
                config.update(json_config_dict)

        return config

    except Exception as e:
        logger.error(f"Error retrieving configuration: {e}")
        return {} 