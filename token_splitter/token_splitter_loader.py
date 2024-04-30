import os

import chromadb
from llama_index.legacy.storage import StorageContext
from llama_index.legacy import ServiceContext, VectorStoreIndex
from llama_index.legacy.vector_stores import ChromaVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.node_parser import TokenTextSplitter


from pdf_parser import PdfMinerParser
from pipeline.llm_ident import giga_llama_llm


def get_program_name_from_file(file_name):
    """
    Получить название программы из названия файла
    :param file_name: название файла
    :return: название программы (ТОЛЬКО ЦИФРЫ)
    """
    return file_name.split('.')[0].split(' ')[1]


def add_lines(nodes):
    cur_page_number = 0
    cur_line = 0
    for node in nodes:
        if node.metadata['page_number'] == cur_page_number:
            node.metadata['start_line'] = cur_line
            node.metadata['end_line'] = cur_line + len(node.text.split('\n')) - 1
            cur_line = cur_line + len(node.text.split('\n'))
        else:
            cur_page_number = node.metadata['page_number']
            cur_line = 0
            node.metadata['start_line'] = 0
            node.metadata['end_line'] = cur_line + len(node.text.split('\n')) - 1
            cur_line = cur_line + len(node.text.split('\n'))
    return nodes


dir = 'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/программы_для_тестов'
files = os.listdir(dir)

parser = PdfMinerParser()

db = chromadb.PersistentClient(path='../VDB_new_splitter')
collection = db.get_or_create_collection(name='token_splitter')

token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')

storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=giga_llama_llm)

if __name__ == '__main__':
    for file_name in files:
        print(f'ОБРАБОТКА ФАЙЛА {file_name}')
        path = f'{dir}/{file_name}'
        documents = parser.parse(path)
        file_parents, file_parents_ids = [], []
        for page in documents:
            page.metadata.pop("bboxes")
        nodes = token_splitter.get_nodes_from_documents(documents=documents)
        nodes = add_lines(nodes=nodes)
        program_name = get_program_name_from_file(file_name=file_name)
        for idx, node in enumerate(nodes):
            node.metadata['program_name'] = program_name
            node.text = node.text.replace('\n', ' ')
            VectorStoreIndex(nodes=[node], storage_context=storage_context, service_context=service_context)
