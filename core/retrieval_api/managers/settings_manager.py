from typing import Dict, Any, Optional
import logging
from .secrets_manager import get_secret, get_config

logger = logging.getLogger(__name__)

class SettingsManager:
    """Manager for application settings and configuration."""
    
    def __init__(self):
        """Initialize the settings manager."""
        self._config = get_config()
        self._secrets = {}
        
    @property
    def get_config(self) -> Dict[str, Any]:
        """Get the application configuration.
        
        Returns:
            Configuration dictionary
        """
        return self._config
        
    @property
    def get_secret(self) -> Dict[str, str]:
        """Get the application secrets.
        
        Returns:
            Secrets dictionary
        """
        return self._secrets
        
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value.
        
        Args:
            key: Setting key
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        return self._config.get(key, default)
        
    def get_secret_value(self, key: str, default: str = None) -> Optional[str]:
        """Get a specific secret value.
        
        Args:
            key: Secret key
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        if key not in self._secrets:
            self._secrets[key] = get_secret(key)
        return self._secrets.get(key, default) 