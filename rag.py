
"""
Поиск релевантной вопроу информации и генерации по ней ответа
"""
import json
import os

import pandas as pd
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.indices import VectorStoreIndex
import chromadb
from llama_index.legacy.vector_stores import ChromaVectorStore, ExactMatchFilter, MetadataFilters
from catboost import CatBoostClassifier
from tqdm import tqdm

from pipeline.llm_ident import giga_llama_llm, giga_langchain_llm_strict
from llama_index.legacy.response_synthesizers import get_response_synthesizer, ResponseMode
from pipeline.prompts import qa_template, refine_template, query_rewriting_no_doc_info_prompt_tmpl
from pipeline import questions


def rewrite_question_no_doc_info(question: str) -> str:
    """
    Переписать вопрос, убрав из него указание на документ (номер программы, описание программы, описание деятельности компании)
    """
    chain = query_rewriting_no_doc_info_prompt_tmpl | giga_langchain_llm_strict
    response = chain.invoke({'question': question}).content
    return response


def get_program_name_from_file(file_name):
    """
    Получить название программы из названия файла
    :param file_name: название файла
    :return: название программы (ТОЛЬКО ЦИФРЫ)
    """
    return file_name.split('.')[0].split(' ')[1]


db = chromadb.PersistentClient(path='db/VDB')
chroma_collection = db.get_collection(name="xops360-2226")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')

service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=giga_llama_llm)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                           service_context=service_context)

llm_response_generator = get_response_synthesizer(service_context=service_context,
                                                  response_mode=ResponseMode.COMPACT,
                                                  text_qa_template=qa_template,
                                                  refine_template=refine_template)


# reranker = CatBoostClassifier().load_model('C:/Users/ADM/Downloads/reranker1to1_prompt2.cbm')

dir = 'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/программы_для_тестов/'
files = os.listdir(dir)
# files = ['ПП 574.pdf']

if __name__ == '__main__':
    for file_name in files:
        file_output = []
        print(f'ПОИСК ПО ДОКУМЕНТУ {file_name}')
        program_name = get_program_name_from_file(file_name=file_name)
        base_retriever = index.as_retriever(similarity_top_k=6,
                                            filters=MetadataFilters(filters=[ExactMatchFilter(key='program_name',
                                                                                              value=program_name)]))
        questions_list = questions.create_question_list(file_name=file_name)
        for question in questions_list:
            print(f'ОБРАБОТКА ВОПРОСА: {question}')
            no_doc_info_question = rewrite_question_no_doc_info(question=question)
            print(f'СВАЛИДИРОВАННЫЙ ВОПРОС: {no_doc_info_question}')
            text, nodes_scores, context = [], [], []
            print('СТАРТ ПОИСКА РЕЛЕВАНТНЫХ ЧАНКОВ')
            retrieved_nodes = base_retriever.retrieve(no_doc_info_question)
            # df_dict = {}
            # for i in range(1536):
            #     df_dict[f'feature {i}'] = []
            # embed_question = embed_model.embed_documents(texts=no_doc_info_question)[0]
            # for node in tqdm(retrieved_nodes):
            #     embed_chunk = embed_model.embed_documents(texts=node.text)[0]
            #     for i in range(768):
            #         df_dict[f'feature {i}'].append(embed_question[i])
            #     for i in range(768, 1536):
            #         df_dict[f'feature {i}'].append(embed_chunk[i % 768])
            # df = pd.DataFrame(data=df_dict)
            # pred_proba = reranker.predict_proba(df)[:, 1]
            # top_ids = pred_proba.argsort()[::-1][:6]
            # reranked_nodes = []
            # for i in range(len(retrieved_nodes)):
            #     if i in top_ids:
            #         reranked_nodes.append(retrieved_nodes[i])
            with open('db/parents_xops360-2226.json', 'r', encoding='utf-8') as f:
                parents = json.load(f)
            for node in retrieved_nodes:
                for parent in parents:
                    if parent['id'] == node.metadata['parent_id']:
                        text.append(parent['text'].replace('\n', ' '))
                        # context.append({'page_number': parent['page_number'],
                        #                 'context_lines': {'start_line': parent['start_line'],
                        #                                   'end_line': parent['end_line']}})
                nodes_scores.append(node.score)
            print('ОТПРАВКА ЗАПРОСА С КОНТЕКСТОМ В МОДЕЛЬ')
            response = llm_response_generator.get_response(query_str=no_doc_info_question, text_chunks=text)
            file_output.append({
                'origin question': question,
                'rewrite question': no_doc_info_question,
                # 'context': context,
                'text': text,
                'response': response,
                'nodes_score': nodes_scores
            })
        print('ЗАПИСЬ В ФАЙЛ')
        with open(f'C:/Users/ADM/OneDrive/Desktop/RAG/XOPS360-2226/{program_name}.json', 'w', encoding="utf-8") as f:
            json.dump(file_output, f, ensure_ascii=False, indent=4)
