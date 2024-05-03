import re
from typing import List, Dict

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from elasticsearch import Elasticsearch
from langchain.embeddings import HuggingFaceEmbeddings

from pdf_parser import PdfMinerParser
from summarize_splitter import summarize_splitter


def text_validation(text: str) -> str:
    stop_words = set(stopwords.words('russian'))
    stemmer = SnowballStemmer("russian")
    punctuation_symbols = [",", "''", " ", "``"]
    word_tokens = word_tokenize(text=text)
    filtered_words = [word for word in word_tokens if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    clean_words = [word for word in stemmed_words if word not in punctuation_symbols]
    return ' '.join(clean_words)


def get_program_name(page_0: str) -> str:
    splits = page_0.split('\n')
    splits.pop(0)
    text = ' '.join(splits)
    start_program_name = text.split('N ')
    program_name = start_program_name[1].split('"')
    return program_name[0]+program_name[1]


EMBED_MODEL = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')


class ElasticSearchConn:
    def __init__(self):
        self._conn = Elasticsearch(hosts='http://localhost:9200')
    # def __enter__(self):
    #     pass

    def create_simple_index(self, index_name: str):
        if self._conn.indices.exists(index=index_name) is True:
            self._conn.indices.delete(index=index_name)
        self._conn.indices.create(index=index_name)

    def create_embedd_index(self, index_name: str):
        if self._conn.indices.exists(index=index_name) is True:
            self._conn.indices.delete(index=index_name)
        self._conn.indices.create(index=index_name, mappings={
            'properties': {
                'embedding': {
                    'type': 'dense_vector',
                    'dims': 768,
                    'index': True,
                    'similarity': 'cosine'
                }
            }
        })

    def add_documents(self, index_name: str, documents: List[Dict]):
        for doc in documents:
            self._conn.index(index=index_name, body=doc)

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     pass


if __name__ == '__main__':
    file = 'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/программы/ПП 26.pdf'
    docs = PdfMinerParser().parse(fpath=file)
    raw_program_name = get_program_name(page_0=docs[0].text)
    print('ПОЛУЧЕНО ВАЛИДНОЕ НАЗВАНИЕ ПРОГРАММЫ')
    valid_program_name = text_validation(text=raw_program_name)
    print('СТАРТ РАЗДЕЛЕНИЯ ТЕКСТА НА ЧАНКИ')
    chunks = summarize_splitter.split(documents=[docs[1]])
    chunks_to_store = []
    for chunk in chunks:
        embedding = EMBED_MODEL.embed_documents(texts=[chunk['text']])[0]
        elastic_document = {
            'embedding': embedding,
            'text': chunk['parent_text'],
            'program_name': valid_program_name,
            'page_number': chunk['page_number']
        }
        chunks_to_store.append(elastic_document)

    es = ElasticSearchConn()
    print('СОЗДАНИЕ ИНДЕКСОВ')
    es.create_simple_index(index_name='programs_name')
    es.create_embedd_index(index_name='main')
    print('ДОБАВЛЕНИЕ ДАННЫХ В БАЗУ')
    es.add_documents(index_name='programs_name', documents=[{'program_name': valid_program_name}])
    es.add_documents(index_name='main', documents=chunks_to_store)

    # es = Elasticsearch(hosts='http://localhost:9200')
    # print(es.indices.exists(index='programs_name'))
    # es.indices.delete(index='programs_name')
    # es.indices.delete(index='main')
#     query = 'Какая ставка дается компаниям, которые участвуют в развитии туризма?'
#     query2 = 'Какая ставка дается компаниям, которые развивают туризм?'
#     query3 = 'Кака ставка дается компаниям по программе 141?'
#     response = es.search(index='program_names', query={'match': {'full_program_name': text_validation(text=query3)}})
#     print(response['hits'])

