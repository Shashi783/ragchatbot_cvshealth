from typing import Optional, List
import tiktoken
import logging

from .llm_manager import LLMManager
# from .managers.prompt_manager import PromptManager
from ..retrievers.compound_retriever import CompoundQueryRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import TreeSummarize

logger = logging.getLogger(__name__)


class ReportCitationQueryEngine(CitationQueryEngine):
    async def _aretrieve_report(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if not isinstance(self.retriever, CompoundQueryRetrieverReport):
            raise TypeError("Retriever must be CompoundQueryRetrieverReport")
        nodes = await self._retriever._aretrieve_report(query_bundle)
        # for postprocessor in self._node_postprocessors:
        #     nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)

        return nodes

    async def get_report_subqueries(self, query_bundle: QueryBundle) -> List[QueryBundle]:
        """Get subqueries for report generation."""
        if not isinstance(self.retriever, CompoundQueryRetrieverReport):
            raise TypeError("Retriever must be CompoundQueryRetrieverReport")
        return await self._retriever.get_report_subqueries(query_bundle)


    # async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    #     nodes = await self._retriever.aretrieve(query_bundle)

    #     for postprocessor in self._node_postprocessors:
    #         nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)

    #     return nodes


class QueryEngineManager:
    def __init__(self, model_manager, prompts, config, cohere_api_key, mode = "CHAT"):
        self._model_manager = model_manager
        self._prompts = prompts
        self._config = config
        self._cohere_api_key = cohere_api_key
        self._mode = mode.upper()  # Store mode as uppercase for consistency

    def build(self, index, metadata_filters: Optional[List[MetadataFilter]] = None) -> CitationQueryEngine:
        """Build and return a fully configured CitationQueryEngine instance."""
        try:
            sim_processor = SimilarityPostprocessor(
                similarity_cutoff=self._config("RAG_SIMILARITY_CUTOFF")
            )

            # TODO: Support LLM reranker option
            rerank = CohereRerank(
                api_key=self._cohere_api_key,
                model=self._config("COHERE_RERANKER"),
                top_n=self._config("RAG_RERANKED_TOP_N"),
            )

            citation_qa_template = PromptTemplate(
                self._prompts.citation_template + self._prompts.qa_template
            )
            citation_refine_template = PromptTemplate(
                self._prompts.citation_template + self._prompts.refine_template
            )

            base_retriever = index.as_retriever(
                similarity_top_k=self._config("RAG_SIMILARITY_TOP_K")
            )
            query_gen_prompt = self._prompts.chat_subquery
            # Choose prompt based on mode
            if self._mode.upper() == "REPORT":
                custom_synthesizer = TreeSummarize(
                    llm=self._model_manager.llm,
                    summary_template=PromptTemplate(self._prompts.report_synthesizer),
                    streaming=True
                )

                query_gen_prompt = self._prompts.chat_subquery

                retriever = CompoundQueryRetrieverReport(
                compound_query_classify_prompt=self._prompts.is_compound_prompt,
                node_postprocessors=[rerank, sim_processor],
                report_subquery_gen_prompt=self._prompts.report_subquery,
                query_gen_prompt=query_gen_prompt,
                retrievers=[base_retriever],
                llm=self._model_manager.llm,
                num_queries=4,
                similarity_top_k=self._config("RAG_SIMILARITY_TOP_K"),
                mode="reciprocal_rerank",
                )

                query_engine = ReportCitationQueryEngine.from_args(
                index,
                retriever=retriever,
                embed_model=self._model_manager.embed_model,
                chat_mode="context",
                citation_chunk_size=self._config("RAG_CITATION_CHUNK_SIZE"),
                citation_chunk_overlap=self._config("RAG_CITATION_CHUNK_OVERLAP"),
                citation_qa_template=citation_qa_template,
                citation_refine_template=citation_refine_template,
                similarity_top_k=self._config("RAG_SIMILARITY_TOP_K"),
                node_postprocessors=[rerank, sim_processor],
                filters=MetadataFilters(filters=metadata_filters or []),
                llm=self._model_manager.llm,
                streaming=self._config("RAG_STREAMING"),
                # response_synthesizer=custom_synthesizer,
                )
            else:
                retriever = CompoundQueryRetriever(
                compound_query_classify_prompt=self._prompts.is_compound_prompt,
                node_postprocessors=[rerank, sim_processor],
                query_gen_prompt=query_gen_prompt,
                retrievers=[base_retriever],
                llm=self._model_manager.llm,
                num_queries=4,
                similarity_top_k=self._config("RAG_SIMILARITY_TOP_K"),
                mode="reciprocal_rerank",
                )
                query_engine = CitationQueryEngine.from_args(
                index,
                retriever=retriever,
                embed_model=self._model_manager.embed_model,
                chat_mode="context",
                citation_chunk_size=self._config("RAG_CITATION_CHUNK_SIZE"),
                citation_chunk_overlap=self._config("RAG_CITATION_CHUNK_OVERLAP"),
                citation_qa_template=citation_qa_template,
                citation_refine_template=citation_refine_template,
                similarity_top_k=self._config("RAG_SIMILARITY_TOP_K"),
                node_postprocessors=[rerank, sim_processor],
                filters=MetadataFilters(filters=metadata_filters or []),
                llm=self._model_manager.llm,
                streaming=self._config("RAG_STREAMING"),
                )

            
            logger.info(f"embedding model {self._model_manager.embed_model}")

            # query_engine = CitationQueryEngine.from_args(
            #     index,
            #     retriever=retriever,
            #     embed_model=self._model_manager.embed_model,
            #     chat_mode="context",
            #     citation_chunk_size=self._config("RAG_CITATION_CHUNK_SIZE"),
            #     citation_chunk_overlap=self._config("RAG_CITATION_CHUNK_OVERLAP"),
            #     citation_qa_template=citation_qa_template,
            #     citation_refine_template=citation_refine_template,
            #     similarity_top_k=self._config("RAG_SIMILARITY_TOP_K"),
            #     node_postprocessors=[rerank, sim_processor],
            #     filters=MetadataFilters(filters=metadata_filters or []),
            #     llm=self._model_manager.llm,
            #     streaming=self._config("RAG_STREAMING"),
            # )

            logger.info("Successfully initialized query engine")

            # Log prompt lengths
            try:
                enc = tiktoken.encoding_for_model(self._model_manager.model_name)
                logger.info(f"Citation QA prompt tokens: {len(enc.encode(str(citation_qa_template)))}")
                logger.info(f"Citation Refine prompt tokens: {len(enc.encode(str(citation_refine_template)))}")
            except Exception as e:
                logger.warning(f"Error calculating token counts: {e}")

            # Log configuration
            logger.info(f"Similarity cutoff: {self._config('RAG_SIMILARITY_CUTOFF')}")
            logger.info(f"Cohere reranker model: {self._config('COHERE_RERANKER')}")
            logger.info(f"Reranked top N: {self._config('RAG_RERANKED_TOP_N')}")
            logger.info(f"Citation chunk size: {self._config('RAG_CITATION_CHUNK_SIZE')}")
            logger.info(f"Citation chunk overlap: {self._config('RAG_CITATION_CHUNK_OVERLAP')}")
            logger.info(f"Similarity top K: {self._config('RAG_SIMILARITY_TOP_K')}")
            logger.info(f"Streaming: {self._config('RAG_STREAMING')}")

            return query_engine

        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")
            raise
