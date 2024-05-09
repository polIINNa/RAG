from typing import List

import chromadb
from llama_index.legacy.storage import StorageContext
from llama_index.legacy import ServiceContext, VectorStoreIndex
from llama_index.legacy.vector_stores import ChromaVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.schema import TextNode
import pandas as pd

from pipeline.pdf_parser import PdfMinerParser
from legacy import summarize_splitter
from pipeline.llm_ident import gpt_llm, giga_llama_llm
from pipeline.prompts import synth_questions_prompt


def create_synth_questions(text: str) -> List[str]:
    chain = synth_questions_prompt | gpt_llm
    response = chain.invoke({'text': text}).content
    raw_qs = response.split('\n')
    questions = []
    for raw_q in raw_qs:
        questions.append(raw_q.split('. ')[1])
    return questions


if __name__ == '__main__':
    dir = '/программы/'
    df_dict = {}
    for i in range(1536):
        df_dict[f'feature {i}'] = []
    df_dict['label'] = []
    files = ['ПП 1570.pdf']
    # files = os.listdir(dir)
    for file in files:
        questionss = []
        print(f'ОБРАБОТКА ФАЙЛА {file}')
        # СОЗДАНИЕ ЧАНКОВ
        print('СОЗДАНИЕ ЧАНКОВ')
        docs = PdfMinerParser().parse(fpath=f'{dir}{file}')
        chunks = summarize_splitter.split(documents=docs)

        # ЗАГРУЗКА ЧАНКОВ В ВЕКТОРНУЮ БАЗУ ДАННЫХ CHROMADB
        print('ЗАГРУЗКА ЧАНКОВ В ВЕКТОРНУЮ БАЗУ ДАННЫХ CHROMADB')
        db = chromadb.PersistentClient(path='db/VDB')
        collection = db.get_or_create_collection(name='my_collection')
        embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
        storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=giga_llama_llm)
        nodes = []
        for chunk in chunks:
            nodes.append(TextNode(text=chunk['text']))
        VectorStoreIndex(nodes=nodes, storage_context=storage_context, service_context=service_context)

        #ГЕНЕРАЦИЯ СИНТ ВОПРОСОВ
        print('ГЕНЕРАЦИЯ СИНТ ВОПРОСОВ')
        for chunk in chunks:
            try:
                questions = create_synth_questions(text=chunk['text'])
                for question in questions:
                    questionss.append({'question': question,
                                       'chunk': chunk['text']})
            except:
                print(f'ПРИ ГЕНЕРАЦИИ/ВАЛИДАЦИИ ВОПРОСОВ ПРОИЗОШЛА ОШИБКА')

        # ОПРЕДЕЛЕНИЕ РЕЛЕВАНТНЫХ И НЕРЕЛЕВАНТНЫХ ЧАНКОВ ДЛЯ ВОПРОСА ЧЕРЕЗ РЕТРИВЕР
        print('ОПРЕДЕЛЕНИЕ РЕЛЕВАНТНЫХ И НЕРЕЛЕВАНТНЫХ ЧАНКОВ ДЛЯ ВОПРОСА ЧЕРЕЗ РЕТРИВЕР')
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                                   service_context=service_context)
        base_retriever = index.as_retriever(similarity_top_k=10)
        markup_dataset = []
        for sample in questionss:
            retrieved_nodes = base_retriever.retrieve(sample['question'])
            retrieved_nodes_markup = []
            for node in retrieved_nodes:
                if node.text.replace('\n', ' ') == sample['chunk'].replace('\n', ' '):
                    retrieved_nodes_markup.append({'node_text': node.text,
                                                   'node_relevancy': 1})
                else:
                    retrieved_nodes_markup.append({'node_text': node.text,
                                                   'node_relevancy': 0})
                markup_dataset.append({'question': sample['question'],
                                        'relevant_chunk': sample['chunk'],
                                        'retrieved_nodes': retrieved_nodes_markup})
            # ЕСЛИ РЕТРИВЕР НАШЕЛ ТОЛЬКО НЕРЕЛЕВАТНЫЕ ЧАНКИ - РУКАМИ ДОБАВЛЯЕМ РЕЛЕВАНТНЫЙ
            print('ПРОВЕРКА ЕСЛИ РЕТРИВЕР НАШЕЛ ТОЛЬКО НЕРЕЛЕВАТНЫЕ ЧАНКИ - РУКАМИ ДОБАВЛЯЕМ РЕЛЕВАНТНЫЙ')
            for sample in markup_dataset:
                scores_sum = 0
                for node in sample['retrieved_nodes']:
                    scores_sum += node['node_relevancy']
                print('ДОБАВЛЕНИЕ РУКАМИ РЕЛЕВАНТНОГО ЧАНКА')
                if scores_sum == 0:
                    sample['retrieved_nodes'].pop()
                    sample['retrieved_nodes'].append({'node_text': sample['relevant_chunk'],
                                                      'node_relevancy': 1})

        # ДОБАВЛЕНИЕ ДАННЫХ В ДАТАФРЕЙМ: СОЗДАНИЕ ЭМБЕДДИНГОВ, СОЗДАНИЕ PANDAS DF И ЗАПИСЬ В CSV
        print('ДОБАВЛЕНИЕ ДАННЫХ В ДАТАФРЕЙМ: СОЗДАНИЕ ЭМБЕДДИНГОВ, СОЗДАНИЕ PANDAS DF И ЗАПИСЬ В CSV')
        for sample in markup_dataset:
            emb_question = embed_model.embed_documents(texts=[sample['question']])[0]
            for chunk in sample['retrieved_nodes']:
                emb_chunk = embed_model.embed_documents(texts=[chunk['node_text']])[0]
                for i in range(768):
                    df_dict[f'feature {i}'].append(emb_question[i])
                for i in range(768, 1536):
                    df_dict[f'feature {i}'].append(emb_question[i % 768])
                df_dict['label'].append(chunk['node_relevancy'])
    # ЗАПИСЬ ДАТАФРЕЙМА В CSV
    print('ЗАПИСЬ ДАТАФРЕЙМА В CSV')
    df = pd.DataFrame(data=df_dict)
    print(df.shape)
    train_id = int(df.shape[0]*(2/3))
    df_train = df.iloc[:train_id]
    df_test = df.iloc[train_id:]
    df_train.to_csv('C:/Users/ADM/Downloads/my_train.csv')
    df_test.to_csv('C:/Users/ADM/Downloads/my_test.csv')


