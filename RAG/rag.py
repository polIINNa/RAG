from typing import List
from pathlib import Path

import chromadb
from aiogram import Bot
from llama_index.legacy.indices import VectorStoreIndex
from llama_index.legacy.schema import TextNode, IndexNode
from llama_index.legacy.node_parser import SentenceSplitter
from llama_index.legacy.retrievers import RecursiveRetriever
from llama_index.legacy.service_context import ServiceContext
from langchain.embeddings import SentenceTransformerEmbeddings
from llama_index.legacy.response_synthesizers import get_response_synthesizer, ResponseMode

from RAG.llm_ident import giga_llama_llm, giga_langchain_llm
from RAG.prompts import qa_template, refine_template, program_name_template


class RAG:
    def __init__(self):
        self._child_nodes_sized = [128, 256, 512]
        self._child_nodes_parsers = [SentenceSplitter(chunk_size=s, chunk_overlap=20) for s in self._child_nodes_sized]
        self._embedding = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self._service_context = ServiceContext.from_defaults(embed_model=self._embedding, llm=giga_llama_llm)
        self._response_synthesizer = get_response_synthesizer(service_context=self._service_context,
                                                              response_mode=ResponseMode.COMPACT,
                                                              text_qa_template=qa_template,
                                                              refine_template=refine_template)
        chromadb_path = Path(__file__).parent / 'gospodderzka_db'
        self._db = chromadb.PersistentClient(path=str(chromadb_path))
        self._chroma_collection = self._db.get_collection(name="main")

    @staticmethod
    def _get_program_name_from_query(query: str) -> str:
        """
        Определить название программы, по которой задается вопрос
        :param query: запрос
        :return: название программы
        """
        chain = program_name_template | giga_langchain_llm
        response = chain.invoke({'query_str': query})
        return response.content

    @staticmethod
    def _create_parent_child_nodes(parent_nodes: List[TextNode], parsers: List[SentenceSplitter]) -> List[IndexNode]:
        """
        Получить список дочерних и родительских IndexNodes (ноды с ссылками на другие объекты)
        :param parent_nodes: список родительских нод (BaseNodes)
        :param parsers: список парсеров, которые разбивают родительские ноды на дочерние
        :return: список родительских и дочерних нод (IndexNodes)
        """
        all_nodes = []
        for parent in parent_nodes:
            for parser in parsers:
                child_nodes = parser.get_nodes_from_documents([parent])
                child_index_nodes = [IndexNode.from_text_node(node, parent.node_id) for node in child_nodes]
                all_nodes.extend(child_index_nodes)
            all_nodes.append(IndexNode.from_text_node(parent, parent.node_id))
        return all_nodes

    async def process(self, query: str, bot: Bot, user_id: int) -> str:
        try:
            program_name_from_query = self._get_program_name_from_query(query=query)
            await bot.send_message(chat_id=user_id, text=f'Ответ буду искать в документе Постановление Правительства №<b>{program_name_from_query}</b>')
            data = self._chroma_collection.get(where={'program_name': program_name_from_query})

            parent_base_nodes = [TextNode(text=data['documents'][i]) for i in range(len(data))]

            all_nodes = self._create_parent_child_nodes(parent_nodes=parent_base_nodes,
                                                        parsers=self._child_nodes_parsers)
            all_nodes_dict = {n.node_id: n for n in all_nodes}
            vector_store_index = VectorStoreIndex(nodes=all_nodes, service_context=self._service_context)
            base_retriever = vector_store_index.as_retriever(similarity_top_k=4)
            recursive_retriever = RecursiveRetriever(
                "vector",
                retriever_dict={"vector": base_retriever},
                node_dict=all_nodes_dict,
                verbose=True,
            )

            await bot.send_message(chat_id=user_id, text='Запуск ретривера')
            nodes = recursive_retriever.retrieve(query)
            context = [node.text for node in nodes]

            await bot.send_message(chat_id= user_id, text='Стучусь в GigaChat, старт генерации ответа')
            response = self._response_synthesizer.get_response(text_chunks=context, query_str=query)

        except Exception:
            response = f'В процессе обработки вопроса {query} произошла ошибка :('
        return response

