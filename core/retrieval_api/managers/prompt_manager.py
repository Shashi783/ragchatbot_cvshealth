import logging
import os
from pathlib import Path
from typing import Dict, Any, Callable
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class PromptConfig(BaseModel):
    """Configuration for prompt templates."""
    greeting_classifier: str = "greeting_classifier.prompt"
    citation_template: str = "citation_template.prompt"
    refine_template: str = "refine_template.prompt"
    qa_template: str = "qa_template.prompt"
    related_queries_template: str = "related_queries.prompt"
    greeting: str = "greeting.prompt"
    profanity_filter: str = "profanity_filter.prompt"
    history_summarizer: str = "history_summarizer.prompt"
    rephrased_query: str = "rephrased_query.prompt"
    system_prompt: str = "system.prompt"

    class Config:
        arbitrary_types_allowed = True

class PromptManager:
    """Manages loading and accessing prompt templates."""
    @staticmethod
    def load_prompts(config: Callable[[str], Any]) -> PromptConfig:
        """Load all prompt templates from the prompts directory."""
        try:
            logger.info("Loading prompt templates...")
            prompts = {}
            prompt_dir = config("PROMPT_DIR")  # Get the PROMPT_DIR value using get_setting
            prompts_parent_dir = os.path.dirname(os.path.dirname(__file__))
            
            print(prompts_parent_dir)
            for prompt_name, prompt_info in PromptConfig.__fields__.items():
                with open(
                    Path(prompts_parent_dir, prompt_dir, prompt_info.default),
                    "r",
                    encoding="utf-8",
                ) as f:
                    prompts[prompt_name] = f.read().strip()

            logger.info("Successfully loaded all prompt templates")
            return PromptConfig(**prompts)

        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise 