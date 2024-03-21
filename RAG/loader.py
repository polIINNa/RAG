import os
from typing import List

from llama_index.legacy.readers import SimpleDirectoryReader
from llama_index.legacy.node_parser import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.legacy import ServiceContext
from llama_index.legacy.storage import StorageContext
from llama_index.legacy.indices import VectorStoreIndex
from llama_index.legacy.schema import Document
from bs4 import BeautifulSoup

from llm_ident import giga_llama_llm
from RAG.vectore_stores import ChromaVS


class Loader:
    def __init__(self, embedding=HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base'),
                 splitter=TokenTextSplitter(chunk_size=512, chunk_overlap=0),
                 vector_store=ChromaVS().chroma_vector_store, llm=giga_llama_llm):
        self._embedding = embedding
        self._splitter = splitter
        self._vector_store = vector_store
        self._llm = llm
        self._service_context = ServiceContext.from_defaults(embed_model=self._embedding, llm=self._llm)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

    @staticmethod
    def _get_program_number_from_file_name(file_name: str) -> str:
        """
        Получиить номер Постановления (название программы) из названия файла
        :param file_name: название файла
        :return: назване программы
        """
        return file_name.split('.')[0].strip('ПП').strip(' ')

    @staticmethod
    def _create_llama_documents(file_path: str, file_name: str, program_number: str) -> List[Document]:
        """
        Получить список Documents - объекты llama-index
        :param file_name: название файла
        :return: список Documents
        """
        file = f'{file_path}/{file_name}'
        extension_name = file_name.split('.')[1]
        if extension_name == 'html':
            with open(file, "r") as f:
                soup = BeautifulSoup(f, 'lxml')
            text = soup.get_text(separator="")
            documents = [Document(text=text)]
        else:
            documents = SimpleDirectoryReader(input_files=[file]).load_data()
        for doc in documents:
            doc.metadata['program_number'] = program_number
        return documents

    def load_documents(self, file_path='/Users/21109090/Desktop/госпрограмма/программы/'):
        """
        Загрузить документы в векторное хранилище
        :return:
        """
        files = os.listdir(file_path)
        for file in files:
            program_number_from_file = self._get_program_number_from_file_name(file)
            documents = self._create_llama_documents(file_path=file_path, file_name=file,
                                                     program_number=program_number_from_file)
            for doc in documents:
                doc.text = doc.text.replace("\n", " ")

            base_nodes = self._splitter.get_nodes_from_documents(documents=documents)

            print(f'ДОБАВЛЕНИЕ НОД ФАЙЛА {file} В ВЕКТОРНОЕ ХРАНИЛИЩЕ')
            VectorStoreIndex(nodes=base_nodes, service_context=self._service_context,
                             storage_context=self._storage_context)
