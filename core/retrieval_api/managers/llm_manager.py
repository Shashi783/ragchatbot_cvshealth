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

    def __init__(self, settings: SettingsManager, mode: str = "chat"):
        """Initialize LLM manager.
        
        Args:
            settings: Settings manager instance
        """
        self._settings = settings
        self._mode = mode.upper()  # Ensure mode is uppercase to match config keys
        self._llm: Optional[LLM] = None
        self._model_name: Optional[str] = None
        self._load_llm(self._mode)

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

    def _load_llm(self, mode = "CHAT") -> None:
        """Load the LLM model based on settings."""
        try:
            model_type = self._settings.get_setting("LLM_MODEL_TYPE", "OPENAI")
            config = self._settings._config.get(model_type, {})
            mode_config = config.get(mode, config)  # fallback to flat config if not nested
            model_name = mode_config.get("MODEL_NAME","gpt-4")
            self._model_name = model_name

            logger.info(f"Loading LLM model: {model_name}, {mode_config}")

            if model_type == "OPENAI":
                self._llm = OpenAI(
                    model=model_name,
                    temperature=mode_config.get("TEMPERATURE", None),
                    max_tokens=mode_config.get("MAX_TOKENS", None),
                    api_key=self._settings.get_secret_value("OPENAI_API_KEY")
                )
            elif model_type == "ANTHROPIC":
                self._llm = Anthropic(
                    model=model_name,
                    temperature=mode_config.get("TEMPERATURE", None),
                    max_tokens=mode_config.get("MAX_TOKENS", None),
                    api_key=self._settings.get_secret_value("ANTHROPIC_API_KEY")
                )
            elif model_type == "OPENROUTER":
                from llama_index.llms.openrouter import OpenRouter
                logger.info(f"Loading OpenRouter LLM {mode} model, max_tokens: {mode_config.get('MAX_TOKENS', None)}")
                self._llm = OpenRouter(
                    api_key=self._settings.get_secret_value("OPENROUTER_API_KEY"),
                    max_tokens=mode_config.get("MAX_TOKENS", None),
                    context_window=mode_config.get("CONTEXT_WINDOW", 8192),
                    model=model_name,
                    stream= True
                )

                # lc_llm = ChatOpenAI(
                # model=model_name,  # or any supported model
                # temperature=self._settings._config.get(model_type, {}).get("TEMPERATURE", None),
                # base_url="https://openrouter.ai/api/v1",
                # api_key=self._settings.get_secret_value("OPENROUTER_API_KEY"),
                # )
                logger.info("no error at OpenRouter")

                # self._llm = LangChainLLM(llm=lc_llm)
            else:
                raise ValueError(f"Unsupported LLM model type: {model_type}")
                
            logger.info(f"Successfully loaded LLM model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
            raise

    def _load_llm_dynamic(self, mode = "CHAT") -> None:
        """Load the LLM model based on settings."""
        try:
            model_type = self._settings.get_setting("LLM_MODEL_TYPE", "OPENAI")
            config = self._settings._config.get(model_type, {})
            mode_config = config.get(mode, config)  # fallback to flat config if not nested
            model_name = mode_config.get("MODEL_NAME","gpt-4")
            self._model_name = model_name

            logger.info(f"Loading LLM model in dynamic: {model_name}, {mode_config}")

            if model_type == "OPENAI":
                self._llm = OpenAI(
                    model=model_name,
                    temperature=mode_config.get("TEMPERATURE", None),
                    max_tokens=mode_config.get("MAX_TOKENS", None),
                    api_key=self._settings.get_secret_value("OPENAI_API_KEY")
                )
            elif model_type == "ANTHROPIC":
                self._llm = Anthropic(
                    model=model_name,
                    temperature=mode_config.get("TEMPERATURE", None),
                    max_tokens=mode_config.get("MAX_TOKENS", None),
                    api_key=self._settings.get_secret_value("ANTHROPIC_API_KEY")
                )
            elif model_type == "OPENROUTER":
                from llama_index.llms.openrouter import OpenRouter
                logger.info(f"Loading OpenRouter LLM {mode} model, max_tokens: {mode_config.get('MAX_TOKENS', None)}")
                self._llm = OpenRouter(
                    api_key=self._settings.get_secret_value("OPENROUTER_API_KEY"),
                    max_tokens=mode_config.get("MAX_TOKENS", None),
                    context_window=mode_config.get("CONTEXT_WINDOW", 8192),
                    model=model_name,
                    stream= True
                )

                # lc_llm = ChatOpenAI(
                # model=model_name,  # or any supported model
                # temperature=self._settings._config.get(model_type, {}).get("TEMPERATURE", None),
                # base_url="https://openrouter.ai/api/v1",
                # api_key=self._settings.get_secret_value("OPENROUTER_API_KEY"),
                # )
                logger.info("no error at OpenRouter")

                # self._llm = LangChainLLM(llm=lc_llm)
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
        model_type = self._settings.get_setting("LLM_MODEL_TYPE", "OPENAI")
        return {
            "model_type": model_type,
            "model_name": self.model_name,
            "temperature": self._settings._config.get(model_type, {}).get("TEMPERATURE", None),
            "max_tokens": self._settings._config.get(model_type, {}).get("MAX_TOKENS", None)
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

    async def acomplete(self, prompt: str) -> str:
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
