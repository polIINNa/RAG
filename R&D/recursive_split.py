import os

import chromadb
from llama_index.legacy.storage import StorageContext
from llama_index.legacy import ServiceContext, VectorStoreIndex
from llama_index.legacy.vector_stores import ChromaVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.legacy.schema import TextNode

from pdf_parser import PdfMinerParser
from llm_ident import giga_llama_llm


def get_program_name_from_file(file_name):
    """
    Получить название программы из названия файла
    :param file_name: название файла
    :return: название программы
    """
    return file_name.split('.')[0].split(' ')[1]


def add_lines_in_metadata(base_nodes):
    """
    Добавить start_line и end_line у чанка
    :param base_nodes:
    :return:
    """
    cur_page_number = 0
    cur_line = 0
    for node in base_nodes:
        if node.metadata['page_number'] == cur_page_number:
            node.metadata['position_start_line'] = cur_line
            node.metadata['position_end_line'] = cur_line + len(node.text.split('\n')) - 1
            cur_line = cur_line + len(node.text.split('\n'))
        else:
            cur_page_number = node.metadata['page_number']
            cur_line = 0
            node.metadata['position_start_line'] = 0
            node.metadata['position_end_line'] = cur_line + len(node.text.split('\n')) - 1
            cur_line = cur_line + len(node.text.split('\n'))
    return base_nodes


dir = '/Users/21109090/Desktop/госпрограмма/программы_для_тестов'
files = os.listdir(dir)

parser = PdfMinerParser()
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=['.', ';', '\n', ' '],
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len
)

db = chromadb.PersistentClient(path='../VDB')
collection = db.get_or_create_collection(name='recursive_split')
embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=giga_llama_llm)

for file_name in files:
    path = f'{dir}/{file_name}'
    documents = parser.parse(path)
    recursive_nodes = []
    program_name = get_program_name_from_file(file_name=file_name)
    for page in documents:
        page.metadata.pop("bboxes")
        doc_nodes_text = recursive_splitter.split_text(page.text)
        for text in doc_nodes_text:
            recursive_nodes.append(TextNode(text=text, metadata={'page_number': page.metadata['page_number'],
                                                                 'program_name': program_name}))

    recursive_nodes = add_lines_in_metadata(recursive_nodes)
    for idx, node in enumerate(recursive_nodes):
        node.id_ = f"node-{program_name}-{idx}"
        node.text = node.text.replace("\n", " ")

    VectorStoreIndex(nodes=recursive_nodes, storage_context=storage_context, service_context=service_context)





