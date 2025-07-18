import logging
import json
import tiktoken  # For token counting
import uuid
import time
import warnings
import re
from pathlib import Path
from typing import Optional, Dict, Generator, List, Any, Union, Tuple, Callable, AsyncGenerator
from pydantic import BaseModel
from urllib.parse import quote
from datetime import datetime

import qdrant_client
import asyncio
import nest_asyncio
from tavily import TavilyClient
from llama_index.core import StorageContext, Settings, load_index_from_storage,PromptHelper,VectorStoreIndex
from llama_index.core.schema import QueryBundle, Node
from llama_index.core.query_engine import CitationQueryEngine

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
# from llama_index.experimental.question_gen.llm_generators import SubQuestionGenerator

# from llama_index.core.question_gen.llm_generators import SubQuestionGenerator
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import TextNode,NodeWithScore
from llama_index.core.postprocessor import LLMRerank

from .managers.storage_manager import StorageManager
from .managers.llm_manager import LLMManager
from .managers.prompt_manager import PromptManager
from .retrievers.compound_retriever import CompoundQueryRetriever

warnings.filterwarnings("ignore")

# Configure environment
nest_asyncio.apply()

logger = logging.getLogger(__name__)


class Generate:
    """Class for generating responses using RAG.
    This class is doing
        1. getting configd and secrets
        2. getting request related data
        3. getting storage manager, LLM manager 
        4. Preparing the query
        7. getting response from query engine
        8. gets responses for is_greeting, is_compoundquery and handle_compund_queries
    """

    def __init__(
        self,
        config: Callable[[str], Any],
        secret: Callable[[str], Any],
        chat_id: str,
        query: str,
        category_name: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
        persist_dir: str = "persist",
        collection_name: str = "rag_llm",
        s3_manager: Optional[Any] = None,
        storage_manager: StorageManager = None,
        llm_manager: LLMManager = None,
        index: VectorStoreIndex = None,
        query_engine: CitationQueryEngine = None,
        is_web_search: bool = True,
    ):
        """Initialize the Generate class."""
        self._config = config
        self._secret = secret
        self._chat_id = chat_id
        self._query = query
        self._category_name = category_name
        self._metadata = metadata or {}
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._s3_manager = s3_manager
        self._llm_manager = llm_manager
        self._is_web_search = is_web_search

        
        # Get API keys from config instead of secrets
        self._tavily_api_key = self._secret("TAVILY_API_KEY")
        self._openai_api_key = self._secret("OPENAI_API_KEY")
        self._anthropic_api_key = self._secret("ANTHROPIC_API_KEY")
        self._cohere_api_key = self._secret("COHERE_API_KEY")

        logger.info(f"Initializing Generate for category: {category_name}")
        self._prompts = PromptManager.load_prompts(self._config)
        self._model_manager = self._llm_manager

        # Load chat history and prepare query
        self._storage_manager = storage_manager
        self.query_engine = query_engine
        # self._index = index


        # self._storage_manager.load_chat_history(chat_id)
        try:
            enc = tiktoken.encoding_for_model(self._model_manager.model_name)  # Or your model's encoding

            chat_hist_tokens = len(enc.encode(str(self._storage_manager._chat_hist)))
            logger.debug(f"chat_hist_tokens, {chat_hist_tokens}")
        except Exception as e:
            logger.info(f"error in printing tokens: {e}")

        # self._refined_query = self._prepare_query()
        self._refined_query = self._query


    def _prepare_query(self) -> str:
        """Prepare the refined query with chat history."""
        if self._storage_manager.chat_hist is not None:
            return f"<|CHAT HISTORY|>: {self._storage_manager.chat_hist}\n\n<|QUERY|>: {self._query}"
        return f"<|QUERY|>: {self._query}"

    async def _is_greeting(self):
        response_ = await self._model_manager.llm.acomplete(
                self._prompts.greeting_classifier.format(query=self._query)
            )
        is_greeting = response_.text.strip()
        logger.info(f"is_greeting, {is_greeting}")
        return is_greeting

    async def _handle_greeting(self):
        logger.info("Query classified as greeting")
        greeting_prompt = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=self._prompts.system_prompt,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=self._prompts.greeting.format(query=self._query),
            ),
        ]
        # greeting_response = self._model_manager.llm.stream_chat(greeting_prompt)
        for text in self._model_manager.llm.stream_chat(greeting_prompt):
            # logger.info(f"Yielding message {text.delta}")  # Debugging log
            yield json.dumps({
                "response_id": str(uuid.uuid4()),
                "type": "greeting",
                "text": text.delta,
            })

    async def generate_answer(self) -> AsyncGenerator[str, None]:
        """Async generator that yields chat responses for the given query."""
        try:
            logger.debug("Checking if query is a greeting")

            if self._model_manager.llm is None:
                logger.error("LLM model is None! Check initialization.")
                raise ValueError("LLM model is not initialized.")
            start_time = time.perf_counter()

            is_greeting = await self._is_greeting()
            logger.info(f"greeting classified at: {time.perf_counter() - start_time}")
            if is_greeting is True:
                logger.info("[Greeting Handler] Entering greeting handler.")
                async for chunk in self._handle_greeting():
                    yield chunk
                return
                
            # Retrieve documents
            retrieved_docs = await self._retrieve_documents()
            logger.info(f"Time taken for document retrieval: {time.perf_counter() - start_time}, number of docs : {len(retrieved_docs)}")


            if not self._has_valid_docs(retrieved_docs):
                if self._is_web_search:
                    async for chunk in self._websearch():
                        yield chunk
                else:
                    for chunk in self._handle_no_retrieval_results():
                        yield chunk
                return

            # Generate response
            logger.info("Generating response...")
            logger.info(f"refined_query {self._refined_query}")
            response = await self.query_engine.asynthesize(
                query_bundle=QueryBundle(query_str=self._refined_query),
                nodes=retrieved_docs,
            )
            

            answer = ""
            first_token_time = None
            if response:
                logger.info(f"asynthesize done at : {time.perf_counter() - start_time}")
                async for text in response.response_gen:
                    if text != "Empty Response":
                        answer += text
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                            logger.info(f"first token generated at : {first_token_time - start_time}")
                    yield json.dumps({
                        "response_id": str(uuid.uuid4()),
                        "type": "tokens",
                        "text": text,
                    })
            else:
                async for chunk in self._handle_no_retrieval_results():
                    yield chunk
                return
            logger.info(f"Answer generated at : {time.perf_counter() - start_time}")
           

            # Process contexts and citations
            contexts, answer = await self._process_contexts(answer, response.source_nodes, retrieved_docs)
            async for chunk in self._yield_context_answer(contexts, answer):
                yield chunk
            
            logger.info(f"processed contexts and citations at : {time.perf_counter() - start_time}")

            # Get Related Queries
            if answer:
                async for chunk in self._get_related_queries(response, answer):
                    yield chunk

            logger.info(f"related queries at : {time.perf_counter() - start_time}")
            # Token counting (optional, keep sync if no I/O involved)
            try:
                enc = tiktoken.encoding_for_model(self._model_manager.model_name)
                answer_tokens = len(enc.encode(str(answer)))
                query_tokens = len(enc.encode(str(self._query)))
                logger.debug(f"Query tokens: {query_tokens}, Answer tokens: {answer_tokens}")
            except Exception as e:
                logger.error(f"Error counting tokens: {e}")

            # Generate conversation title if no chat history
            try:
                if self._storage_manager.chat_hist is None:
                    async for chunk in self._generate_conversation_title():
                        yield chunk
                    logger.info(f"conversation title generated at : {time.perf_counter() - start_time}")
            except Exception as e:
                logger.critical(f"Error generating conversation title: {e}")
            
            # Update chat history
            self._storage_manager.chat_hist = f"{self._refined_query}\n{answer}\n\n"
            logger.info("Successfully completed response generation")

        except asyncio.CancelledError:
            logger.warning("Client disconnected, stopping response generation.")
            return

        except Exception as e:
            logger.critical(f"Error generating answer: {e}")
            raise

    def _handle_no_retrieval_results(self):
        logger.info("landed into no retieval results")
        # Yield a message suggesting the user to refine their query
        yield json.dumps({
            "response_id": str(uuid.uuid4()),
            "type": "answer",
            "text": (
                "I couldn’t find any relevant information based on your query in the current knowledge base. "
                "Please try rephrasing your question or include more specific details to help improve the results."
            )
        })
        # Yield a context type response with an empty context and prompt user for more details
        yield json.dumps({
            "response_id": str(uuid.uuid4()),
            "type": "context",
            "text": json.dumps({})  # No context found
        })

    async def _retrieve_documents(self):
        logger.info("Retrieving relevant documents...")
        retrieved_docs = await self.query_engine.aretrieve(QueryBundle(query_str=self._refined_query))
        logger.info(f"Score of retrieved docs: {[doc.score for doc in retrieved_docs]}")
        return retrieved_docs

    def _has_valid_docs(self, docs):
        if docs and max([doc.score for doc in docs]) > self._config("RAG_SIMILARITY_CUTOFF"):
            return True
        else:
            return False

    # def _decompose_query(self, query: str) -> List[str]:
    #     try:
    #         chat_sub_query_prompt = self._prompts.chat_subquery.format(query =query)
    #         response = self._model_manager.llm.complete(chat_sub_query_prompt).text.strip()
    #         subqueries = self._parse_subqueries(response)
    #         if isinstance(subqueries, list) and all(isinstance(q, str) for q in subqueries):
    #             return subqueries
    #         else:
    #             raise ValueError("Invalid format: not a list of strings")
    #     except json.JSONDecodeError as e:
    #         raise ValueError(f"Subquery response not valid JSON: {e}")
    
    async def _websearch(self):
        logger.warning("No relevant contexts retrieved")
        if self._is_web_search != "True":
            yield json.dumps({
                "response_id": str(uuid.uuid4()),
                "type": "answer",
                "text": "No relevant contexts retrieved",
            })
            return

        search_results = self._tavily_client.search(
            self._query, max_results=3, search_depth="advanced"
        )["results"]
        content = "\n\n".join([
            f"{idx+1}. Title: {result['title']}\nContent: {result['content']}\nURL: {result['url']}"
            for idx, result in enumerate(search_results)
        ])
        tavily_prompt = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=self._prompts.system_prompt,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=self._prompts.tavily_template.format(
                    search_results=content, query=self._query
                ),
            ),
        ]

        tavily_resp = self._model_manager.llm_model.stream_chat(tavily_prompt)

        for text in tavily_resp:
            yield json.dumps({
                "response_id": str(uuid.uuid4()),
                "type": "tokens",
                "text": text.delta,
            })
 

    def _is_compound_query(self, query: str) -> bool:
        is_compound_prompt = self._prompts.is_compound_prompt.format(query=query)
        resp = self._model_manager.llm.complete(is_compound_prompt)
        return "yes" in resp.text.lower()
    
    def _parse_subqueries(self, llm_output: str) -> list[str]:
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            raise ValueError(f"Subquery output is not valid JSON:\n{llm_output}")

    def _decompose_query(self, query: str) -> List[str]:
        try:
            chat_sub_query_prompt = self._prompts.chat_subquery.format(query =query)
            response = self._model_manager.llm.complete(chat_sub_query_prompt).text.strip()
            subqueries = self._parse_subqueries(response)
            if isinstance(subqueries, list) and all(isinstance(q, str) for q in subqueries):
                return subqueries
            else:
                raise ValueError("Invalid format: not a list of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Subquery response not valid JSON: {e}")

    async def _process_contexts(
        self, answer: str, source_nodes: List[Node], retrieved_docs: List[Node]
    ) -> Dict[str, Dict[str, Any]]:
        """Process and format context information from retrieved documents."""
        try:
            logger.info("Processing context information...")
            extract_pattern = r"^Source \d+:\s*\n"
            cited_nums = re.findall(r"\[(\d+)\]", answer)
            source_lst = []

            for source in source_nodes:
                logger.debug(f"source text, {source.node.get_text()}")
                source_text = re.sub(
                    extract_pattern, "", source.node.get_text(), flags=re.MULTILINE
                ).strip()
                source_lst.append(source_text)
            
            # logger.info(f"sources_list, {source_lst}")

            logger.debug(f"cited_nums, {cited_nums}")

            logger.debug(f"got {len(source_lst)} sources")
            contexts = {}
            retrieved_counter = 0

            for idx, doc in enumerate(retrieved_docs):
                if str(idx + 1) not in cited_nums:
                    logger.debug(f"{str(idx+1)} not in {cited_nums}")
                    continue
                
                if doc.text.strip() in source_lst:
                    retrieved_counter += 1
                    contexts[str(retrieved_counter)] = {
                        "doc_name": doc.metadata["doc_name"],
                        "page_num": doc.metadata["page_number"],
                        # "chunk": doc.metadata["highlighted_chunk"],
                    }
                    answer = answer.replace(
                        f"[{str(idx+1)}]",
                        f'[[{retrieved_counter}]](<{doc.metadata["s3_url"]}>)',
                    )
            logger.debug(f"Processed {contexts} context ")
            # logger.info(f"Processed {answer} answer ")
            return contexts, answer

        except Exception as e:
            logger.error(f"Error processing contexts: {e}")
            raise e
        
    async def _yield_context_answer(self, contexts, answer):
        logger.info(f"yielding contexts, {contexts}")
        logger.debug(f"answer yielding, {answer}")
        yield json.dumps({
            "response_id": str(uuid.uuid4()),
            "type": "answer",
            "text": answer
        })
        yield json.dumps({
            "response_id": str(uuid.uuid4()),
            "type": "context",
            "text": json.dumps(contexts),
        })

    async def _get_related_queries(self, response, answer):
        
        firstfivesources = "\n\n".join(doc.node.get_text() for doc in response.source_nodes[:5])  
        logger.info("generation of related queries started")
        related_queries = self._model_manager.llm.complete(
            self._prompts.related_queries_template.format(
                query=self._query,
                sources=firstfivesources,
                answer=answer,
            ),max_tokens=512
        ).text.strip()

        yield json.dumps({
            "response_id": str(uuid.uuid4()),
            "type": "related",
            "text": related_queries,
        })
        logger.info("generation of related queries ended")
    
    async def _generate_conversation_title(self):
        try:
            logger.debug("Generating conversation title...")
            conversation_title = self._model_manager.llm.complete(
                self._prompts.conv_title_template.format(
                    query=self._query, category=self._category_name
                )
            ).text.strip()

            logger.info("conversation title generated")
            yield json.dumps({
                "response_id": str(uuid.uuid4()),
                "type": "title",
                "text": conversation_title,
            })
        except:
            logger.error(f"Failed to generate conversation title: {e}")
            yield json.dumps({
                "response_id": str(uuid.uuid4()),
                "type": "title",
                "text": "Untitled Conversation"
            })
