from __future__ import annotations

from typing import List, Dict, Tuple
import json

from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from llama_index.legacy.vector_stores import ChromaVectorStore, MetadataFilters, ExactMatchFilter
from llama_index.legacy import ServiceContext, VectorStoreIndex

from RAG.pipeline.llm_interface import gigachat
from RAG.pipeline.prompts_templates import QUERY_REWRITING_NO_DOC_INFO_TMPL, QA_TMPL, PROGRAM_NUMBER_TMPL


class RAG:
    """ RAG - класс для поиска контекста по вопросу и генерации ответа """
    def __init__(self):
        self.embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
        self.db_conn = chromadb.PersistentClient(path='RAG/db/VDB')
        self.collection = self.db_conn.get_or_create_collection(name='main')
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.service_context = ServiceContext.from_defaults(embed_model=self.embed_model, llm=gigachat)
        self.search_index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store,
                                                               service_context=self.service_context)

    @staticmethod
    def _get_program_number_from_query(query: str) -> str:
        """
        Получить номер программы господдержки из вопроса
        :param query: вопрос пользователя
        :return: номер программы господдержки
        """
        #TODO: подумать, может определять номер программы господдержки регуляркой, а не LLM. У  меня в голове нет кейсов, где в вопросе есть прочие цифры, кроме номера программы
        prompt = PromptTemplate.from_template(template=PROGRAM_NUMBER_TMPL)
        chain = prompt | gigachat
        return chain.invoke({'query': query}).content

    @staticmethod
    def _get_program_name_from_query(query: str) -> str:
        """
        Получить название/описание программы господдержки из вопроса
        Пример: Какая льготная ставка дается по программе развития туризма? Ответ: программа по развитию туризма
        :param query: вопрос пользователя
        :return: название/описание программы господдержки
        """
        return '-1'

    @staticmethod
    def _map_program_name(program_name_from_query: str) -> str | None:
        #TODO: подумать над название get_db_program_name()
        """
        Смэтчить название программы господдержки из вопроса с названием программы, которое хранится в базе данных
        :param program_name_from_query: название программы господдержки, полученное из вопроса
        :return: название программы господдержки, которые хранится в базе данных
        """
        pass

    @staticmethod
    def _rewrite_question_no_doc_info(query: str) -> str:
        """
        Переписать вопроса пользователя, убрав из него информацию про документ с программой господдержки
        :param query: вопрос пользователя
        :return: переписанный вопрос
        """
        prompt = PromptTemplate.from_template(template=QUERY_REWRITING_NO_DOC_INFO_TMPL)
        chain = prompt | gigachat
        return chain.invoke({'query': query}).content

    def _retrieve(self, query: str, filters: Dict[str, str], k: int = 6) -> List[str]:
        """
        Найти контекст, в котором содержится релевантная вопросу информация
        :param query: вопрос пользователя
        :param filters: фильтры по метадате чанков
        :return: найденные чанки (контекст)
        """
        exact_match_filters = []
        for key in filters:
            exact_match_filters.append(ExactMatchFilter(key=key, value=filters[key]))
        retriever = self.search_index.as_retriever(similarity_top_k=k,
                                                   filters=MetadataFilters(filters=exact_match_filters))
        retrieved_nodes = retriever.retrieve(query)
        with open('RAG/db/parents.json', 'r') as f:
            parents = json.load(f)
        extracted_passages = []
        for node in retrieved_nodes:
            for parent in parents:
                if parent['id'] == node.metadata['parent_id']:
                    extracted_passages.append(parent['text'].replace('\n', ' '))
        return extracted_passages

    @staticmethod
    def _generate_answer(query: str, extracted_passages: List[str]) -> str:
        """
        Сгенерировать вопрос
        :param query: вопрос пользователя
        :param extracted_passages: найденные ретриверов чанки (контекст)
        :return: ответ на вопрос
        """
        text = '.'.join(extracted_passages)
        prompt = PromptTemplate.from_template(template=QA_TMPL)
        chain = prompt | gigachat
        return chain.invoke({'context': text, 'query': query}).content

    def get_gs_program(self, query: str) -> Tuple[str, bool] | None:
        """
        Получить программу господдержки, в которой искать ответ на вопрос: номер программы или название/описание
        :param query: вопрос пользователя
        :return: кортеж с определенным номером или названием программы господдержки и флагом про то, есть ли данная программы в бд или нет
        """
        with open('RAG/available_programs.json', 'r') as f:
            available_programs = json.load(f)
        program_number = self._get_program_number_from_query(query=query)
        if program_number != '-1':
            if program_number not in available_programs['available_program_numbers']:
                return program_number, False
            else:
                return program_number, True
        else:
            program_name_from_query = self._get_program_name_from_query(query=query)
            if program_name_from_query != '-1':
                program_name_from_db = self._map_program_name(program_name_from_query=program_name_from_query)
                if program_name_from_db is None:
                    return program_name_from_query, False
                else:
                    return program_name_from_db, True

    def rag(self, query: str, gs_program_name: str) -> str:
        """
        Основная функция класса RAG - поиск контекста и генерация по нему ответа на вопрос
        :param query: вопрос пользователя
        :return: ответ на вопрос
        """
        if gs_program_name.isdigit() is True:
            filters = {'program_number': gs_program_name}
        else:
            filters = {'program_name': gs_program_name}
        no_doc_info_query = self._rewrite_question_no_doc_info(query=query)
        extracted_passages = self._retrieve(query=no_doc_info_query, filters=filters)
        return self._generate_answer(query=no_doc_info_query, extracted_passages=extracted_passages)
