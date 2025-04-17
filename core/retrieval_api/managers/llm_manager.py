## Things to change
# what if OLLAMA is used from an api call?

import logging
from typing import Dict, Any, Union, Optional, List, Generator
from tenacity import retry, stop_after_attempt, wait_exponential
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import LLM
from ..models.chat import ChatMessage
from .settings_manager import SettingsManager
# print(dir(llama_index.core.llms))

# from llama_index.core.llms.types import BaseLLM

# from llama_index.llms import BaseLLM

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages loading and accessing LLM models."""

    _embed_model = None
    _llm_model = None

    def __init__(self, settings: SettingsManager):
        """Initialize LLM manager.
        
        Args:
            settings: Settings manager instance
        """
        self._settings = settings
        self._llm: Optional[LLM] = None
        self._model_name: Optional[str] = None
        self._load_llm()

    def _load_embed_model(self) -> HuggingFaceEmbedding:
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self._settings.get_setting('HF_EMBED')}")
            return HuggingFaceEmbedding(model_name=self._settings.get_setting('HF_EMBED'))
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _load_llm(self) -> None:
        """Load the LLM model based on settings."""
        try:
            model_type = self._settings.get_setting("LLM_MODEL_TYPE", "OPENAI")
            model_name = self._settings.get_setting("LLM_MODEL_NAME", "gpt-4")
            self._model_name = model_name
            
            logger.info(f"Loading LLM model: {model_name}")
            
            if model_type == "OPENAI":
                self._llm = OpenAI(
                    model=model_name,
                    temperature=0.7,
                    max_tokens=4096,
                    api_key=self._settings.get_secret_value("OPENAI_API_KEY")
                )
            elif model_type == "ANTHROPIC":
                self._llm = Anthropic(
                    model=model_name,
                    temperature=0.7,
                    max_tokens=4096,
                    api_key=self._settings.get_secret_value("ANTHROPIC_API_KEY")
                )
            else:
                raise ValueError(f"Unsupported LLM model type: {model_type}")
                
            logger.info(f"Successfully loaded LLM model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
            raise

    @property
    def model_name(self) -> str:
        """Get the name of the loaded model.
        
        Returns:
            str: Name of the loaded model
        """
        if not self._model_name:
            raise ValueError("No model loaded")
        return self._model_name
    
    @property
    def llm(self) -> LLM:
        """Get the loaded LLM model.
        
        Returns:
            BaseLLM: The loaded LLM model
            
        Raises:
            ValueError: If no model is loaded
        """
        if not self._llm:
            raise ValueError("No LLM model loaded")
        return self._llm

    def get_model_config(self) -> Dict[str, Any]:
        """Get the configuration for the current model.
        
        Returns:
            Dict[str, Any]: Model configuration
        """
        return {
            "model_type": self._settings.get_setting("LLM_MODEL_TYPE", "OPENAI"),
            "model_name": self.model_name,
            "temperature": 0.7,
            "max_tokens": 4096
        }

    @property
    def embed_model(self) -> HuggingFaceEmbedding:
        """Get the embedding model."""
        if self._embed_model is None:
            self._embed_model = self._load_embed_model()
        return self._embed_model

    @property
    def llm_model(self) -> Union[OpenAI, Ollama, Anthropic]:
        """Get the LLM model."""
        if self._llm_model is None:
            self._llm_model = self._load_llm()
        return self._llm_model

    async def generate_response(self, message: str) -> str:
        """Generate a response using the LLM model.
        
        Args:
            message: The input message
            
        Returns:
            str: The generated response
        """
        try:
            response = await self._llm.acomplete(message)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def complete(self, prompt: str) -> str:
        """Complete a prompt using the LLM model.
        
        Args:
            prompt: The input prompt
            
        Returns:
            str: The completed text
        """
        try:
            response = self._llm.complete(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error completing prompt: {str(e)}")
            raise

    def stream_chat(self, messages: List[ChatMessage]) -> Generator[str, None, None]:
        """Stream chat responses using the LLM model.
        
        Args:
            messages: List of chat messages
            
        Yields:
            str: Response chunks
        """
        try:
            for chunk in self._llm.stream_chat(messages):
                yield chunk.delta
        except Exception as e:
            logger.error(f"Error streaming chat: {str(e)}")
            raise 