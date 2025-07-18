import logging
from enum import Enum
import json
from typing import Dict, List, Optional, Tuple, cast
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
# from llama_index.core.constants import FUSION_MODES
from llama_index.core.postprocessor.types import BaseNodePostprocessor

logger = logging.getLogger(__name__)


class FUSION_MODES(str, Enum):
    """Enum for different fusion modes."""

    RECIPROCAL_RANK = "reciprocal_rerank"  # apply reciprocal rank fusion
    RELATIVE_SCORE = "relative_score"  # apply relative score fusion
    DIST_BASED_SCORE = "dist_based_score"  # apply distance-based score fusion
    SIMPLE = "simple"  # simple re-ordering of results based on original scores


class CompoundQueryRetriever(QueryFusionRetriever):
    def __init__(self, compound_query_classify_prompt: str, node_postprocessors: List[BaseNodePostprocessor], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compound_query_classify_prompt = compound_query_classify_prompt
        self._node_postprocessors = node_postprocessors

    async def _is_compound_query(self, query: str) -> bool:
        is_compound_prompt = self.compound_query_classify_prompt.format(query=query)
        resp = await self._llm.acomplete(is_compound_prompt)
        return "yes" in resp.text.lower()

    def _parse_subqueries(self, llm_output: str) -> list[str]:
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            raise ValueError(f"Subquery output is not valid JSON:\n{llm_output}")

    async def _decompose_query(self, query: str) -> List[str]:
        try:
            sub_query_gen_prompt = self.query_gen_prompt.format(query =query)
            response = await self._llm.acomplete(sub_query_gen_prompt)
            subqueries = self._parse_subqueries(response.text)
            if isinstance(subqueries, list) and all(isinstance(q, str) for q in subqueries):
                return subqueries
            else:
                raise ValueError("Invalid format: not a list of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Subquery response not valid JSON: {e}")

    async def _get_queries(self, original_query: str) -> List[QueryBundle]:
        # Check if the query is compound
        subqueries = []
        if await self._is_compound_query(original_query):
            subqueries = await self._decompose_query(original_query)
        # The LLM often returns more queries than we asked for, so trim the list.
        logger.info(f"Subqueries: {subqueries}")
        return [QueryBundle(q) for q in subqueries[: self.num_queries - 1]]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)

        if self.mode == FUSION_MODES.RECIPROCAL_RANK:
            return self._reciprocal_rerank_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.RELATIVE_SCORE:
            return self._relative_score_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.DIST_BASED_SCORE:
            return self._relative_score_fusion(results, dist_based=True)[
                : self.similarity_top_k
            ]
        elif self.mode == FUSION_MODES.SIMPLE:
            return self._simple_fusion(results)[: self.similarity_top_k]
        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")
            
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(await self._get_queries(query_bundle.query_str))

        results = await self._run_async_queries(queries)

        if self.mode == FUSION_MODES.RECIPROCAL_RANK:
            return self._reciprocal_rerank_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.RELATIVE_SCORE:
            return self._relative_score_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.DIST_BASED_SCORE:
            return self._relative_score_fusion(results, dist_based=True)[
                : self.similarity_top_k
            ]
        elif self.mode == FUSION_MODES.SIMPLE:
            return self._simple_fusion(results)[: self.similarity_top_k]
        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")