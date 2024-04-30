import os
import json
from typing import List

import chromadb
from llama_index.legacy.storage import StorageContext
from llama_index.legacy import ServiceContext, VectorStoreIndex
from llama_index.legacy.vector_stores import ChromaVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.schema import TextNode


from pdf_parser import PdfMinerParser
from pipeline.llm_ident import giga_llama_llm
from subpoints_splitter import subpoints_spitter


def get_program_name_from_file(file_name):
    """
    Получить название программы из названия файла
    :param file_name: название файла
    :return: название программы (ТОЛЬКО ЦИФРЫ)
    """
    return file_name.split('.')[0].split(' ')[1]


def add_lines(parents_datas: List):
    cur_page_number = 0
    cur_line = 0
    parents_datas = sorted(parents_datas, key=lambda x: x['page_number'])
    for parent_data in parents_datas:
        if parent_data['page_number'] == cur_page_number:
            parent_data['start_line'] = cur_line
            parent_data['end_line'] = cur_line + len(parent_data['text'].split('\n')) - 1
            cur_line = cur_line + len(parent_data['text'].split('\n'))
        else:
            cur_page_number = parent_data['page_number']
            cur_line = 0
            parent_data['start_line'] = 0
            parent_data['end_line'] = cur_line + len(parent_data['text'].split('\n')) - 1
            cur_line = cur_line + len(parent_data['text'].split('\n'))
    return parents_datas


dir = 'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/программы_для_тестов'
files = os.listdir(dir)

parser = PdfMinerParser()

db = chromadb.PersistentClient(path='../VDB_new_splitter')
collection = db.get_or_create_collection(name='subpoints')

embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')

storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=giga_llama_llm)
parents, chunks, nodes = [], [], []
if __name__ == '__main__':
    for file_name in files:
        print(f'ОБРАБОТКА ФАЙЛА {file_name}')
        program_name = get_program_name_from_file(file_name=file_name)
        path = f'{dir}/{file_name}'
        documents = parser.parse(path)
        file_parents, file_parents_ids = [], []
        for page in documents:
            page.metadata.pop("bboxes")
        file_chunks = subpoints_spitter.split(documents=documents)
        # Создаем список родительских чанков файла
        for chunk in file_chunks:
            if chunk['parent_id'] not in file_parents_ids:
                parent = {'id': chunk['parent_id'],
                          'page_number': chunk['page_number'],
                          'text': chunk['parent_text']
                          }
                file_parents.append(parent)
                file_parents_ids.append(chunk['parent_id'])
            nodes.append(TextNode(text=chunk['text'], metadata={'page_number': chunk['page_number'],
                                                                'parent_id': chunk['parent_id'],
                                                                'program_name': program_name}))
        # Добавляем номера строк начала и конца родительских чанков
        file_parents = add_lines(parents_datas=file_parents)
        # Добавляем в список всех родительских чанков по всем файлам
        parents.extend(file_parents)
        chunks.extend(file_chunks)

    # Сохраняем данные
    for node in nodes:
        node.text = node.text.replace('\n', ' ')
        VectorStoreIndex(nodes=[node], storage_context=storage_context, service_context=service_context)
    with open('parents.json', 'w', encoding='utf-8') as f:
        json.dump(parents, f, indent=4)
    with open('search_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=4)

