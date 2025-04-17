import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from .settings_manager import SettingsManager
from .log_manager import LogManager
from .prompt_manager import PromptManager
from .llm_manager import LLMManager

logger = logging.getLogger(__name__)

class AppManager:
    """Manages FastAPI application and its dependencies."""

    _settings: Optional[SettingsManager] = None
    _llm = None
    _prompts = None
    _app: Optional[FastAPI] = None

    def __init__(self):
        if AppManager._settings is None:
            AppManager._settings = SettingsManager()
            LogManager.setup_logging(
                AppManager._settings.get_config,
                AppManager._settings.get_secret
            )

    @property
    def settings(self) -> SettingsManager:
        """Get the settings manager."""
        return AppManager._settings

    @property
    def app(self) -> FastAPI:
        """Get or create the FastAPI application."""
        if AppManager._app is None:
            logger.info("Creating FastAPI application")

            AppManager._app = FastAPI(
                title="AltGAN RAG API",
                version="1.0",
                description="API for Retrieval-Augmented Generation system",
            )

            AppManager._app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        return AppManager._app

    @property
    def prompts(self):
        """Get or load the prompts."""
        if AppManager._prompts is None:
            AppManager._prompts = PromptManager.load_prompts(
                AppManager._settings.get_config
            )

        return AppManager._prompts

    @property
    def llm(self):
        """Get or initialize the LLM."""
        if AppManager._llm is None:
            AppManager._llm = LLMManager.init_llm(
                AppManager._settings.get_config,
                AppManager._settings.get_secret
            )

        return AppManager._llm 