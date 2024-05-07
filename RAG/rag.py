from llama_index.legacy.indices import VectorStoreIndex
from llama_index.legacy.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.legacy.service_context import ServiceContext
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.response_synthesizers import get_response_synthesizer, ResponseMode

from RAG.vectore_stores import ChromaVS
from RAG.pipeline.llm_ident import giga_llama_llm, giga_langchain_llm
from RAG.pipeline.prompts import qa_template, refine_template, program_name_template


class RAG:
    """Класс с ретривером и генератором ответа"""
    def __init__(self, embedding=HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base'),
                 llm=giga_llama_llm, vector_store=ChromaVS().chroma_vector_store, response_mode=ResponseMode.COMPACT):

        self._embedding = embedding
        self._llm = llm
        self._response_mode = response_mode
        self._vector_store = vector_store
        self._service_context = ServiceContext.from_defaults(embed_model=self._embedding, llm=self._llm)
        self._response_synthesizer = get_response_synthesizer(service_context=self._service_context,
                                                              response_mode=self._response_mode,
                                                              text_qa_template=qa_template,
                                                              refine_template=refine_template)
        self._index = VectorStoreIndex.from_vector_store(vector_store=self._vector_store,
                                                         service_context=self._service_context)

    @staticmethod
    def get_program_number(query) -> str:
        """
        Получить из вопроса номер программы постановления
        :param query: запрос
        :return: номер программы
        """
        chain = program_name_template | giga_langchain_llm
        response = chain.invoke({'query_str': query})
        return response.content

    async def process(self, query: str) -> str:
        try:
            program_number = self.get_program_number(query=query)
            base_retriever = self._index.as_retriever(similarity_top_k=6,
                                                      filters=MetadataFilters(filters=[ExactMatchFilter(key='program_name',
                                                                                                        value=program_number)]))
            retrieved_nodes = base_retriever.retrieve(query)
            context = [node.text for node in retrieved_nodes]
            response = self._response_synthesizer.get_response(text_chunks=context, query_str=query)
        except Exception:
            response = 'В процессе обработки запроса произошла ошибка :(('
        return response

