import os
from typing import List

import chromadb
from bs4 import BeautifulSoup
from llama_index.legacy import ServiceContext
from llama_index.legacy.schema import Document
from llama_index.legacy.storage import StorageContext
from llama_index.legacy.indices import VectorStoreIndex
from llama_index.legacy.readers import SimpleDirectoryReader
from llama_index.legacy.node_parser import SentenceSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from llama_index.legacy.vector_stores import ChromaVectorStore

from RAG.llm_ident import giga_llama_llm

EMBEDDING = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
SERVICE_CONTEXT = ServiceContext.from_defaults(embed_model=EMBEDDING, llm=giga_llama_llm)


def _get_program_name_from_file_name(file_name: str):
    """
    Получить номер программы постановления из названия файла
    :param file_name: Название файла
    :return: Номер постановления
    """
    return file_name.split('.')[0].strip('ПП').strip(' ')


def _create_documents(file_name: str) -> List[Document]:
    """
    Создать объекты Documents
    :param file_name: Название файла
    :return: Объекты Documents, созданные из файла
    """
    file = f'{dir}/{file_name}'
    extension_name = file_name.split('.')[1]
    if extension_name == 'html':
        with open(file, "r") as f:
            soup = BeautifulSoup(f, 'lxml')
        text = soup.get_text(separator="")
        documents = [Document(text=text)]
    else:
        documents = SimpleDirectoryReader(input_files=[file]).load_data()
    for doc in documents:
        doc.metadata['program_name'] = file_name.split('.')[0].strip('ПП').strip(' ')
    return documents


# подключение к бд
db = chromadb.PersistentClient(path="gospodderzka_db")
chroma_collection = db.get_or_create_collection(name="main")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=EMBEDDING, llm=giga_llama_llm)

splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

dir = '/Users/21109090/Desktop/госпрограмма/программы'
files = os.listdir(dir)
for file in files:
    print(f'СТАРТ ЗАГРУЗКИ ФАЙЛА {file} В ВЕКТОРНОЕ ХРАНИЛИЩЕ')
    program_name_from_file = _get_program_name_from_file_name(file)
    documents = _create_documents(file_name=file)
    base_nodes = splitter.get_nodes_from_documents(documents=documents)

    for idx, node in enumerate(base_nodes):
        node.id_ = f"base-{program_name_from_file}-{idx}"
    for node in base_nodes:
        node.metadata['program_name'] = program_name_from_file
    print(f'ДОБАВЛЕНИЕ НОД ФАЙЛА {file} В ВЕКТОРНОЕ ХРАНИЛИЩЕ')
    index = VectorStoreIndex(nodes=base_nodes, service_context=service_context, storage_context=storage_context)
