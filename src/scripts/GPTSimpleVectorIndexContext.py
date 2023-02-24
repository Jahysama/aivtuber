from llama_index.data_structs.data_structs import SimpleIndexDict
from llama_index.indices.base import DOCUMENTS_INPUT
from llama_index.indices.query.vector_store.simple import GPTSimpleVectorIndexQuery
from llama_index.indices.vector_store.base import BaseGPTVectorStoreIndex
from llama_index.prompts.prompts import QuestionAnswerPrompt

from typing import (
    Sequence,
    Type,
)

from llama_index.indices.query.base import BaseGPTIndexQuery
from llama_index.langchain_helpers.text_splitter import TextSplitter
from llama_index.schema import BaseDocument

from typing import Any, Dict, List, Optional, Union, cast

from llama_index.data_structs.data_structs import IndexStruct
from llama_index.docstore import DocumentStore
from llama_index.embeddings.base import BaseEmbedding
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.indices.query.base import BaseQueryRunner
from llama_index.indices.query.query_transform import BaseQueryTransform
from llama_index.indices.query.schema import QueryBundle, QueryConfig, QueryMode
from llama_index.indices.registry import IndexRegistry
from llama_index.langchain_helpers.chain_wrapper import LLMPredictor
from llama_index.response.schema import Response



# TMP: refactor query config type
QUERY_CONFIG_TYPE = Union[Dict, QueryConfig]


class QueryRunnerContext(BaseQueryRunner):
    """Tool to take in a query request and perform a query with the right classes.

    Higher-level wrapper over a given query.

    """

    def __init__(
        self,
        llm_predictor: LLMPredictor,
        prompt_helper: PromptHelper,
        embed_model: BaseEmbedding,
        docstore: DocumentStore,
        index_registry: IndexRegistry,
        query_configs: Optional[List[QUERY_CONFIG_TYPE]] = None,
        query_transform: Optional[BaseQueryTransform] = None,
        recursive: bool = False,
        use_async: bool = False,
    ) -> None:
        """Init params."""
        config_dict: Dict[str, QueryConfig] = {}
        if query_configs is None or len(query_configs) == 0:
            query_config_objs: List[QueryConfig] = []
        elif isinstance(query_configs[0], Dict):
            query_config_objs = [
                QueryConfig.from_dict(cast(Dict, qc)) for qc in query_configs
            ]
        else:
            query_config_objs = [cast(QueryConfig, q) for q in query_configs]

        for qc in query_config_objs:
            config_dict[qc.index_struct_type] = qc

        self._config_dict = config_dict
        self._llm_predictor = llm_predictor
        self._prompt_helper = prompt_helper
        self._embed_model = embed_model
        self._docstore = docstore
        self._index_registry = index_registry
        self._query_transform = query_transform or BaseQueryTransform()
        self._recursive = recursive
        self._use_async = use_async

    def _get_query_kwargs(self, config: QueryConfig) -> Dict[str, Any]:
        """Get query kwargs.

        Also update with default arguments if not present.

        """
        query_kwargs = {k: v for k, v in config.query_kwargs.items()}
        if "prompt_helper" not in query_kwargs:
            query_kwargs["prompt_helper"] = self._prompt_helper
        if "llm_predictor" not in query_kwargs:
            query_kwargs["llm_predictor"] = self._llm_predictor
        if "embed_model" not in query_kwargs:
            query_kwargs["embed_model"] = self._embed_model
        return query_kwargs

    def query(
        self, query_str_or_bundle: Union[str, QueryBundle], index_struct: IndexStruct
    ) -> list:
        """Run query."""
        # NOTE: Currently, query transform is only run once
        # TODO: Consider refactor to support index-specific query transform
        if isinstance(query_str_or_bundle, str):
            query_bundle = self._query_transform(query_str_or_bundle)
        else:
            query_bundle = query_str_or_bundle

        index_struct_type = index_struct.get_type()
        if index_struct_type not in self._config_dict:
            config = QueryConfig(
                index_struct_type=index_struct_type, query_mode=QueryMode.DEFAULT
            )
        else:
            config = self._config_dict[index_struct_type]
        mode = config.query_mode

        query_cls = self._index_registry.type_to_query[index_struct_type][mode]
        # if recursive, pass self as query_runner to each individual query
        query_runner = self if self._recursive else None
        query_kwargs = self._get_query_kwargs(config)
        query_obj = query_cls(
            index_struct,
            **query_kwargs,
            query_runner=query_runner,
            docstore=self._docstore,
            use_async=self._use_async,
        )

        return query_obj.get_nodes_and_similarities_for_response(query_bundle)


class GPTSimpleVectorIndexContext(BaseGPTVectorStoreIndex[SimpleIndexDict]):
    """GPT Simple Vector Index.

    The GPTSimpleVectorIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a simple dictionary.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within the dict.

    During query time, the index uses the dict to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
    """

    index_struct_cls = SimpleIndexDict

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[SimpleIndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        text_splitter: Optional[TextSplitter] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            text_splitter=text_splitter,
            **kwargs,
        )

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTSimpleVectorIndexQuery,
            QueryMode.EMBEDDING: GPTSimpleVectorIndexQuery,
        }

    def _add_document_to_index(
        self,
        index_struct: SimpleIndexDict,
        document: BaseDocument,
    ) -> None:
        """Add document to index."""
        nodes = self._get_nodes_from_document(document)

        id_node_embed_tups = self._get_node_embedding_tups(
            nodes, set(index_struct.nodes_dict.keys())
        )
        for new_id, node, text_embedding in id_node_embed_tups:
            index_struct.add_node(node, text_id=new_id)
            index_struct.add_to_embedding_dict(new_id, text_embedding)

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        text_ids_to_delete = set()
        int_ids_to_delete = set()
        for text_id, int_id in self.index_struct.id_map.items():
            node = self.index_struct.nodes_dict[int_id]
            if node.ref_doc_id != doc_id:
                continue
            text_ids_to_delete.add(text_id)
            int_ids_to_delete.add(int_id)

        for int_id, text_id in zip(int_ids_to_delete, text_ids_to_delete):
            del self.index_struct.nodes_dict[int_id]
            del self.index_struct.id_map[text_id]
            del self.index_struct.embedding_dict[text_id]

    def query(
        self,
        query_str: Union[str, QueryBundle],
        mode: str = QueryMode.DEFAULT,
        query_transform: Optional[BaseQueryTransform] = None,
        use_async: bool = False,
        **query_kwargs: Any,
    ) -> list:
        """Answer a query.

        When `query` is called, we query the index with the given `mode` and
        `query_kwargs`. The `mode` determines the type of query to run, and
        `query_kwargs` are parameters that are specific to the query type.

        For a comprehensive documentation of available `mode` and `query_kwargs` to
        query a given index, please visit :ref:`Ref-Query`.


        """
        mode_enum = QueryMode(mode)
        if mode_enum == QueryMode.RECURSIVE:
            # TODO: deprecated, use ComposableGraph instead.
            if "query_configs" not in query_kwargs:
                raise ValueError("query_configs must be provided for recursive mode.")
            query_configs = query_kwargs["query_configs"]
            query_runner = QueryRunnerContext(
                self._llm_predictor,
                self._prompt_helper,
                self._embed_model,
                self._docstore,
                self._index_registry,
                query_configs=query_configs,
                query_transform=query_transform,
                recursive=True,
                use_async=use_async,
            )
            return query_runner.query(query_str, self._index_struct)
        else:
            self._preprocess_query(mode_enum, query_kwargs)
            # TODO: pass in query config directly
            query_config = QueryConfig(
                index_struct_type=self._index_struct.get_type(),
                query_mode=mode_enum,
                query_kwargs=query_kwargs,
            )
            query_runner = QueryRunnerContext(
                self._llm_predictor,
                self._prompt_helper,
                self._embed_model,
                self._docstore,
                self._index_registry,
                query_configs=[query_config],
                query_transform=query_transform,
                recursive=False,
                use_async=use_async,
            )
            return query_runner.query(query_str, self._index_struct)
