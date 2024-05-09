"""
Загрузчик чанаков в БД.
По чанкам строятся эмбеддинги и кладутся в векторную БД, а родители кладутся в JSON
"""

import os
import json

import chromadb
from llama_index.legacy.storage import StorageContext
from llama_index.legacy import ServiceContext, VectorStoreIndex
from llama_index.legacy.vector_stores import ChromaVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.schema import TextNode

from pipeline.llm_ident import giga_llama_llm


def get_program_name_from_file(file_name):
    """
    Получить название программы из названия файла
    :param file_name: название файла
    :return: название программы (ТОЛЬКО ЦИФРЫ)
    """
    return file_name.split('.')[0].split(' ')[1]


# def add_lines(parents_datas):
#     cur_page_number = 0
#     cur_line = 0
#     parents_datas = sorted(parents_datas, key=lambda x: x['page_place'])
#     for parent_data in parents_datas:
#         if parent_data['page_number'] == cur_page_number:
#             parent_data['start_line'] = cur_line
#             parent_data['end_line'] = cur_line + len(parent_data['text'].split('\n')) - 1
#             cur_line = cur_line + len(parent_data['text'].split('\n'))
#         else:
#             cur_page_number = parent_data['page_number']
#             cur_line = 0
#             parent_data['start_line'] = 0
#             parent_data['end_line'] = cur_line + len(parent_data['text'].split('\n')) - 1
#             cur_line = cur_line + len(parent_data['text'].split('\n'))
#     return parents_datas


db = chromadb.PersistentClient(path='db/VDB')
collection = db.get_or_create_collection(name='xops360-2226')
embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=giga_llama_llm)

if __name__ == '__main__':
    parents, nodes = [], []
    dir = 'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/программы_для_тестов'
    files = os.listdir(dir)
    for file_name in files:
        print(f'ЗАГРУЗКА ЧАНКОВ ДЛЯ ФАЙЛА {file_name}')
        program_name = get_program_name_from_file(file_name=file_name)
        # file_parents = []
        with open(f'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/chunks_xops360-2226/remove_application/{file_name.split(".")[0]}.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        #Создание чанков и родителей
        for chunk in chunks:
            parent = {'id': chunk['parent_id'],
                      'text': chunk['parent_text']}
            node = TextNode(text=chunk['text'], metadata={'program_name': program_name,
                                                          'parent_id': chunk['parent_id']})
            parents.append(parent)
            nodes.append(node)

        # for chunk in chunks:
        #     parent = {
        #               'id': chunk['parent_id'],
        #               'page_number': chunk['page_number'],
        #               'page_place': chunk['parent_id'].split('.pdf-')[1],
        #               'text': chunk['parent_text']
        #               }
        #     all_nodes.append(TextNode(text=chunk['text'], metadata={'page_number': chunk['page_number'],
        #                                                             'parent_id': chunk['parent_id'],
        #                                                             'program_name': program_name}))
        #     file_parents.append(parent)
        # file_parents = add_lines(parents_datas=file_parents)
        # all_parents.extend(file_parents)

    print('СТАРТ ЗАГРУЗКИ ДАННЫХ В ХРАНИЛИЩЕ')
    for node in nodes:
        VectorStoreIndex(nodes=[node], storage_context=storage_context, service_context=service_context)
    with open('db/parents_xops360-2226.json', 'w', encoding='utf-8') as f:
        json.dump(parents, f, indent=4)
