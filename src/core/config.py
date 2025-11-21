import yaml
import os
from typing import Any, Dict
from pathlib import Path

class ConfigLoader:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Assuming the config file is at the root 'config/settings.yaml'
        # Relative to this file (src/core/config.py), it would be ../../config/settings.yaml
        base_path = Path(__file__).parent.parent.parent
        config_path = base_path / "config" / "settings.yaml"

        if not config_path.exists():
            # Fallback or default values if file is missing, though ideally we should raise error
            print(f"Warning: Config file not found at {config_path}. Using defaults.")
            self._config = {}
            return

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

# Global instance access
config = ConfigLoader()
