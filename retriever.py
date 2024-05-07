import json
import os

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.indices import VectorStoreIndex
import chromadb
from llama_index.legacy.vector_stores import ChromaVectorStore, ExactMatchFilter, MetadataFilters

from pipeline.llm_ident import giga_llama_llm


def get_program_name_from_file(file_name):
    """
    Получить название программы из названия файла
    :param file_name: название файла
    :return: название программы (ТОЛЬКО ЦИФРЫ)
    """
    return file_name.split('.')[0].split(' ')[1]


db = chromadb.PersistentClient(path='db/VDB')
chroma_collection = db.get_collection(name="summarize")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')

service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=giga_llama_llm)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                           service_context=service_context)

if __name__ == '__main__':
    dir = 'chunks_questions'
    files = os.listdir(dir)
    for file_name in files:
        print(f'ГЕНЕРАЦИЯ ДАТАСЕТА ДЛЯ РЕРАНКЕРА ДЛЯ ФАЙЛА {file_name}')
        rerank_file_dataset = []
        program_name = get_program_name_from_file(file_name=file_name)
        base_retriever = index.as_retriever(similarity_top_k=10,
                                            filters=MetadataFilters(filters=[ExactMatchFilter(key='program_name',
                                                                                              value=program_name)]))
        with open(f'{dir}/{file_name}', 'r', encoding='utf-8') as f:
            data = json.load(f)
        file_questions = []
        for chunk in data:
            if chunk['questions'] != None:
                for question in chunk['questions']:
                    retrieved_nodes = base_retriever.retrieve(question)
                    output_data = []
                    for node in retrieved_nodes:
                        if node.text.replace('\n', ' ') == chunk['chunk'].replace('\n', ' '):
                            output_data.append({'node_text': node.text,
                                                'node_relevancy': 1,
                                                'node_page': node.metadata['page_number']})
                        else:
                            output_data.append({'node_text': node.text,
                                                'node_relevancy': 0,
                                                'node_page': node.metadata['page_number']})
                    rerank_file_dataset.append({'question': question,
                                                'relevant_chunk': chunk['chunk'],
                                                'data': output_data})
        with open(f'synth_data/{file_name}', 'w', encoding='utf-8') as f:
            json.dump(rerank_file_dataset, f, indent=4)
