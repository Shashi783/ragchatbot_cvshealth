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
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core.schema import QueryBundle, Node
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

from .managers.storage_manager import StorageManager
from .managers.llm_manager import LLMManager
from .managers.prompt_manager import PromptManager

warnings.filterwarnings("ignore")

# Configure environment
nest_asyncio.apply()

logger = logging.getLogger(__name__)

class Generate:
    """Class for generating responses using RAG."""
    
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
        is_web_search: bool = True
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

        # llm_manager = LLMManager(settings_manager)
        
        # Configure LlamaIndex settings
        Settings.llm = self._model_manager.llm_model
        Settings.embed_model = self._model_manager.embed_model

        # Load chat history and prepare query
        self._storage_manager = storage_manager
        self._storage_manager.load_chat_history(chat_id)
        self._refined_query = self._prepare_query()

        # Setup query engine
        self._persist = self._storage_manager.load_persist_dir(persist_dir, collection_name)
        index = self._setup_storage_context(collection_name)
        metadata_filters = self._prepare_metadata_filters(metadata)
        self._init_query_engine(index, metadata_filters)

    def _prepare_query(self) -> str:
        """Prepare the refined query with chat history."""
        if self._storage_manager.chat_hist is not None:
            return f"<|CHAT HISTORY|>: {self._storage_manager.chat_hist}\n\n<|QUERY|>: {self._query}"
        return f"<|QUERY|>: {self._query}"

    def _setup_storage_context(self, collection_name: str) -> StorageContext:
        """Setup storage context with Qdrant vector store."""
        try:
            client = qdrant_client.QdrantClient(
                url=self._secret("QDRANT_URL"), api_key=self._secret("QDRANT_API_KEY")
            )

            aclient = qdrant_client.AsyncQdrantClient(
                url=self._secret("QDRANT_URL"), api_key=self._secret("QDRANT_API_KEY")
            )

            
            vector_store = QdrantVectorStore(
                client=client,
                aclient=aclient,
                collection_name=collection_name,
                enable_hybrid=self._config("QDRANT_ENABLE_HYBRID"),
                fastembed_sparse_model=self._config("FASTEMBED_SPARSE_MODEL"),
                prefer_grpc=False,
            )

            logger.info("client and aclient and vector_store objects are created")


            storage_context = StorageContext.from_defaults(
                persist_dir=self._persist, vector_store=vector_store
            )

            # storage_context = StorageContext.from_defaults(
            #     vector_store=vector_store
            # )

            logger.info("storage context created")

            index = load_index_from_storage(storage_context)
            return index
        except Exception as e:
            logger.error(f"Error setting up storage context: {e}")
            raise

    def _prepare_metadata_filters(self, metadata: Dict[str, Any]) -> List[MetadataFilter]:
        """Prepare metadata filters from metadata dict."""
        if metadata is None:
            logger.warning("No metadata provided; returning empty filter list.")
            return []  # Return an empty list if no metadata is provided
        logger.info(f"Preparing metadata filters: {metadata}")
        return [MetadataFilter(key=key, value=value) for key, value in metadata.items()]

    def _init_query_engine(self, index, metadata_filters: Optional[List[MetadataFilter]]) -> None:
        """Initialize the citation query engine."""
        try:
            sim_processor = SimilarityPostprocessor(
                similarity_cutoff=self._config("RAG_SIMILARITY_CUTOFF")
            )
            rerank = CohereRerank(
                api_key=self._cohere_api_key,
                model=self._config("COHERE_RERANKER"),
                top_n=self._config("RAG_RERANKED_TOP_N"),
            )

            citation_qa_template=PromptTemplate(
                    self._prompts.citation_template + self._prompts.qa_template
                )
            citation_refine_template=PromptTemplate(
                    self._prompts.citation_template + self._prompts.refine_template
                )

            self.query_engine = CitationQueryEngine.from_args(
                index,
                embed_model=self._model_manager.embed_model,
                chat_mode="context",
                citation_chunk_size=self._config("RAG_CITATION_CHUNK_SIZE"),
                citation_chunk_overlap=self._config("RAG_CITATION_CHUNK_OVERLAP"),
                citation_qa_template=citation_qa_template,
                citation_refine_template=citation_refine_template,
                similarity_top_k=self._config("RAG_SIMILARITY_TOP_K"),
                node_postprocessors=[rerank, sim_processor],
                filters=MetadataFilters(filters=metadata_filters or []),
                llm=self._model_manager.llm_model,
                streaming=self._config("RAG_STREAMING"),
            )
            logger.info("Successfully initialized query engine")
        
            # Log prompt lengths
            enc = tiktoken.encoding_for_model(self._model_manager.model_name)  # Or your model's encoding

            qa_prompt_tokens = len(enc.encode(str(citation_qa_template)))
            refine_prompt_tokens = len(enc.encode(str(citation_refine_template)))

            logger.info(f"Citation QA prompt tokens: {qa_prompt_tokens}")
            logger.info(f"Citation Refine prompt tokens: {refine_prompt_tokens}")

            # Log node postprocessor configuration
            logger.info(f"Similarity cutoff: {self._config('RAG_SIMILARITY_CUTOFF')}")
            logger.info(f"Cohere reranker model: {self._config('COHERE_RERANKER')}")
            logger.info(f"Reranked top N: {self._config('RAG_RERANKED_TOP_N')}")

            # Log query engine configuration
            logger.info(f"Citation chunk size: {self._config('RAG_CITATION_CHUNK_SIZE')}")
            logger.info(f"Citation chunk overlap: {self._config('RAG_CITATION_CHUNK_OVERLAP')}")
            logger.info(f"Similarity top K: {self._config('RAG_SIMILARITY_TOP_K')}")
            logger.info(f"Streaming: {self._config('RAG_STREAMING')}")


        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")
            raise

    def generate_answer(self) -> Generator[str, None, None]:
        """Generate and yield the answer for the given query."""
        try:
            logger.debug("Checking if query is a greeting")
            if self._model_manager.llm is None:
                logger.error("LLM model is None! Check initialization.")
                raise ValueError("LLM model is not initialized.")
            is_greeting = self._model_manager.llm.complete(
                self._prompts.greeting_classifier.format(query=self._query)
            ).text.strip()

            if is_greeting == "True":
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
                    print(f"Yielding message {text.delta}")  # Debugging log
                    yield json.dumps({
                        "response_id": str(uuid.uuid4()),
                        "type": "greeting",
                        "text": text.delta,
                    })
                return

            answer = ""
            logger.info("Retrieving relevant documents...")
            # print(print("Stored Documents:", self.query_engine.index.docstore.docs.keys()))
            retrieved_docs = self.query_engine.retrieve(
                QueryBundle(query_str=self._refined_query)
            )

            # logger.info(f"Retrieved documents: {retrieved_docs}")
            logger.info(f"Score of retrieved docs: {[doc.score for doc in retrieved_docs]}")
            for idx, doc in enumerate(retrieved_docs):
                logger.info(f"Document {idx+1}: {doc.node.get_text()}")

            if not retrieved_docs or max([doc.score for doc in retrieved_docs]) <= self._config("RAG_SIMILARITY_CUTOFF"):
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

                return

            # Generate response
            logger.info("Generating response...")
            start_response = time.perf_counter()
            response = self.query_engine.query(self._refined_query)
            end_response = time.perf_counter()
            logger.info(f"Time taken to generate response: {end_response - start_response} seconds")

            for text in response.response_gen:
                if text != "Empty Response":
                    answer += text
                    yield json.dumps({
                        "response_id": str(uuid.uuid4()),
                        "type": "tokens",
                        "text": text,
                    })

            # Process contexts and citations
            logger.info("Processing contexts and citations...")
            contexts, answer = self._process_contexts(answer, response.source_nodes, retrieved_docs)
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

            # Generate related queries
            logger.info("Generating related queries...")
            related_queries = self._model_manager.llm_model.complete(
                self._prompts.related_queries_template.format(
                    query=self._query,
                    sources="\n\n".join(doc.node.get_text() for doc in response.source_nodes),
                    answer=answer,
                )
            ).text.strip()

            yield json.dumps({
                "response_id": str(uuid.uuid4()),
                "type": "related",
                "text": related_queries,
            })

            # Generate conversation title if no chat history
            if self._storage_manager.chat_hist is None:
                logger.info("Generating conversation title...")
                conversation_title = self._model_manager.llm_model.complete(
                    self._prompts.conv_title_template.format(
                        query=self._query, category=self._category_name
                    )
                ).text.strip()

                yield json.dumps({
                    "response_id": str(uuid.uuid4()),
                    "type": "title",
                    "text": conversation_title,
                })

            # Update chat history
            self._storage_manager.chat_hist = f"{self._refined_query}\n{answer}\n\n"
            logger.info("Successfully completed response generation")
        
        except asyncio.CancelledError:
            logger.warning("Client disconnected, stopping response generation.")
            return

        except Exception as e:
            logger.critical(f"Error generating answer: {e}")
            raise


    def _process_contexts(
        self, answer: str, source_nodes: List[Node], retrieved_docs: List[Node]
    ) -> Dict[str, Dict[str, Any]]:
        """Process and format context information from retrieved documents."""
        try:
            logger.debug("Processing context information...")
            extract_pattern = r"^Source \d+:\s*\n"
            cited_nums = re.findall(r"\[(\d+)\]", answer)
            source_lst = []

            for source in source_nodes:
                source_text = re.sub(
                    extract_pattern, "", source.node.get_text(), flags=re.MULTILINE
                ).strip()
                source_lst.append(source_text)

            contexts = {}
            retrieved_counter = 0

            for idx, doc in enumerate(retrieved_docs):
                if str(idx + 1) not in cited_nums:
                    continue

                if doc.text.strip() in source_lst:
                    retrieved_counter += 1
                    contexts[str(retrieved_counter)] = {
                        "file_name": doc.metadata["file_name"],
                        "page_num": doc.metadata["page_num"],
                        "chunk": doc.metadata["highlighted_chunk"],
                    }
                    answer = answer.replace(
                        f"[{str(idx+1)}]",
                        f'[[{retrieved_counter}]]({self._config("PDF_BASE_URL")}{quote(doc.metadata["file_name"])}.pdf)',
                    )

            logger.debug(f"Processed {retrieved_counter} context citations")
            return contexts, answer

        except Exception as e:
            logger.error(f"Error processing contexts: {e}")
            raise 