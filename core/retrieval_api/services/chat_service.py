import logging
from typing import Optional, Dict, Any, Generator
from datetime import datetime
from fastapi.responses import StreamingResponse, JSONResponse
import uuid
from ..models.base import ChatMessage, ChatHistory
from ..schemas.api import ChatRequest, ChatResponse
from ..schemas.rag import ChatRAGRequest
from ..managers.storage_manager import LocalStorageManager
from ..managers.llm_manager import LLMManager
from ..generate import Generate
import time
import json

logger = logging.getLogger(__name__)

class ChatService:
    """Service for handling chat operations."""
    
    def __init__(self, storage_manager: LocalStorageManager, llm_manager: LLMManager, settings_manager):
        self._storage = storage_manager
        self._llm = llm_manager
        self._settings = settings_manager
        
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process a basic chat request."""
        try:
            # Load chat history
            self._storage.load_chat_history(request.chat_id)
            
            # Generate response
            response_text = await self._llm.generate_response(request.message)
            
            # Update chat history
            self._storage.chat_hist = f"{request.message}\n{response_text}\n\n"
            
            return ChatResponse(
                chat_id=request.chat_id,
                message=response_text,
                metadata={"model": self._llm.model_name}
            )
            
        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            raise

    async def process_rag_chat(self, request: ChatRequest) -> StreamingResponse:
        """Process a RAG-based chat request."""
        try:
            start_time = time.perf_counter()
            logger.info(f"Processing RAG chat request for chat_id: {request.chat_id}")
            
            # Load chat history
            self._storage.load_chat_history(request.chat_id)

            
            # Create Generate instance
            generate = Generate(
                config=self._settings.get_setting,
                secret=self._settings.get_secret_value,
                chat_id=request.chat_id,
                query=request.message,
                category_name=request.metadata.get("category", "general"),
                metadata=request.metadata,
                persist_dir="v6_rag",
                collection_name="v6_rag",
                s3_manager=self._storage,
                storage_manager=self._storage,
                llm_manager=self._llm,
                is_web_search=False
            )


            logger.info("starting response generation")
            
            # Generate response
            # response_text = generate.generate_answer()
            # # async for chunk in generate.generate_answer():  # Await the async generator
            # #     yield chunk  # Yield each chunk as it is generated

            # async def stream_response(generate):
            #     async for message in generate.generate_answer():
            #         yield message

            for chunk in generate.generate_answer():
                chunk_data = json.loads(chunk)  # Assuming generate.generate_answer() returns JSON
                chunk_data["chat_id"] = request.chat_id
                chunk_data["metadata"] = {
                    "model": self._llm.model_name,
                    "sources": chunk_data.get("text", []) if chunk_data["type"] == "context" else []
                }
                yield json.dumps(chunk_data)  # Yield the modified JSON chunk
                # yield chunk
            
            # print(response_text.)
            # if response_text is not None:
            #     duration = time.perf_counter() - start_time
            #     logger.info(f"Response generation completed in {duration:.2f} seconds")
            # return StreamingResponse(content=stream_response(generate), media_type="text/event-stream")

            # response_text = ""
            # async for chunk in generate.generate_answer():
            #     chunk_data = json.loads(chunk)
            #     if chunk_data["type"] == "tokens":
            #         response_text += chunk_data["text"]
            
            # # Update chat history
            # self._storage.chat_hist = f"{request.message}\n{response_text}\n\n"
            
            # return ChatResponse(
            #     chat_id=request.chat_id,
            #     message=response_text,
            #     metadata={
            #         "model": self._llm.model_name,
            #         "sources": chunk_data.get("text", []) if chunk_data["type"] == "context" else []
            #     }
            # )
            
        except Exception as e:
            logger.error(f"Error processing RAG chat request: {e}")
            raise

    async def get_chat_history(self, chat_id: str) -> Optional[ChatHistory]:
        """Get chat history for a given chat ID."""
        try:
            # Load chat history
            history_text = self._storage.load_chat_history(chat_id)
            if not history_text:
                logger.info(f"No chat history found for chat_id: {chat_id}")
                return None
                
            # Parse chat history
            messages = []
            for line in history_text.split("\n"):
                if line.strip():
                    if line.startswith("User:"):
                        messages.append(ChatMessage(
                            role="user",
                            content=line[5:].strip(),
                            metadata={"model": self._llm.model_name}
                        ))
                    elif line.startswith("Assistant:"):
                        messages.append(ChatMessage(
                            role="assistant",
                            content=line[10:].strip(),
                            metadata={"model": self._llm.model_name}
                        ))
                        
            if not messages:
                logger.info(f"No valid messages found in chat history for chat_id: {chat_id}")
                return None
                        
            return ChatHistory(
                chat_id=chat_id,
                messages=messages,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            raise

    async def save_chat_history(self, chat_id: str, chat_hist: str, config: Dict[str, Any]) -> None:
        """Background task for chat history summarization and saving to storage.
        
        Args:
            chat_id: ID of the chat to summarize
            chat_hist: Chat history to summarize
            config: Configuration dictionary containing storage settings
        """
        try:
            start_time = time.perf_counter()
            logger.info(f"Starting chat history summarization for chat {chat_id}")

            # Generate chat summary
            logger.debug("Generating chat summary")
            summarized_hist = self._llm.complete(
                self._llm.prompts.history_summarizer.format(chat_history=chat_hist)
            ).text.strip()

            # Save to storage
            logger.debug("Saving summary to storage")
            self._storage.save_chat_summary(chat_id, summarized_hist)

            duration = time.perf_counter() - start_time
            logger.info(
                f"Chat history summarization completed for chat {chat_id} in {duration:.2f} seconds"
            )

        except Exception as e:
            logger.error(f"Error during chat history summarization for chat {chat_id}: {e}")
            raise 