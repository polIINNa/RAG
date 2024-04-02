import os

from llama_index.legacy.node_parser import TokenTextSplitter
import chromadb
from llama_index.legacy.storage import StorageContext
from llama_index.legacy import ServiceContext, VectorStoreIndex
from llama_index.legacy.vector_stores import ChromaVectorStore
from langchain.embeddings import HuggingFaceEmbeddings


from pdf_parser import PyMuPDFParserWithLimit
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

parser = PyMuPDFParserWithLimit()
token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)

db = chromadb.PersistentClient(path='VDB')
collection = db.get_or_create_collection(name='main')
embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=giga_llama_llm)

for file_name in files:
    path = f'{dir}/{file_name}'
    documents = parser.parse(path)
    for page in documents:
        page.metadata.pop("bboxes")

    base_nodes = token_splitter.get_nodes_from_documents(documents=documents)
    base_nodes = add_lines_in_metadata(base_nodes)
    program_name = get_program_name_from_file(file_name=file_name)
    for idx, node in enumerate(base_nodes):
        node.metadata['program_name'] = program_name
        node.id_ = f"node-{program_name}-{idx}"
        node.text = node.text.replace("\n", " ")

    VectorStoreIndex(nodes=base_nodes, storage_context=storage_context, service_context=service_context)

