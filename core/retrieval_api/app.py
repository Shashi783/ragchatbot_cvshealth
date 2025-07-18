## Change chat history summarizer to use local location rather than s3

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import StreamingResponse, JSONResponse
# from pydantic import BaseModel, Field, field_validator
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List,AsyncGenerator
import boto3
import logging
import time
import atexit
from pathlib import Path
from botocore.exceptions import ClientError
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import sys
from dotenv import load_dotenv

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)
# from generate import Generate
# from core.retrieval_api.managers.storage_manager import LocalStorageManager
# from core.retrieval_api.managers.settings_manager import SettingsManager
# from core.retrieval_api.managers import AppManager, PromptManager
# from core.retrieval_api.secrets_manager import get_secret
# # from core.retrieval_api.schemas.rag import RAG
# from core.retrieval_api.services.chat_service import ChatService


from core.retrieval_api.services.chat_service import ChatService
from core.retrieval_api.managers.storage_manager import LocalStorageManager
from core.retrieval_api.managers.llm_manager import LLMManager
from core.retrieval_api.managers.settings_manager import SettingsManager

from core.retrieval_api.models.base import ChatMessage, ChatHistory
from core.retrieval_api.schemas.api import ChatRequest, ChatResponse
from core.retrieval_api.schemas.rag import ChatRAGRequest


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Constants
CONFIG_PATH = Path("config/config.yaml")

start_time = time.perf_counter()

# Initialize settings manager
settings_manager = SettingsManager()

# Initialize managers
storage_manager = LocalStorageManager(settings_manager.get_config)
llm_manager_chat = LLMManager(settings_manager, mode="CHAT")  # Use "chat" mode for chat service

# Initialize chat service
chat_service = ChatService(storage_manager, llm_manager_chat, settings_manager)
# report_service = ChatService(storage_manager, llm_manager_report, settings_manager)

app = FastAPI(
    title="AltGAN API",
    version="1.0",
    description="API for AltGAN RAG system",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/health", tags=["Health Check"])
async def health_check() -> JSONResponse:
    """Basic health check endpoint."""
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "OK"})

@app.post("/v1/chat", tags=["Chat API"])
async def get_answer(request: ChatRequest, background_tasks: BackgroundTasks) -> StreamingResponse:
    """Process chat requests and generate RAG-based responses."""
    print(request.dict())
    try:
        # Process the chat request using the service
        response_generator = chat_service.process_rag_chat(request)

        async def stream_results() -> AsyncGenerator[bytes, None]:
            """Helper function to stream response chunks."""
            async for chunk in response_generator:
                yield chunk.encode('utf-8')

        # # Schedule chat history summarization
        # background_tasks.add_task(
        #     chat_service.save_chat_history, 
        #     request.chat_id, 
        #     storage_manager.chat_hist, 
        #     settings_manager.get_config
        # )
        # Return streaming response
        return StreamingResponse(content=stream_results(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=False,
        workers=1,
    )