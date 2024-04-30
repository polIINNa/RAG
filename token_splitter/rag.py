import json
import os

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.indices import VectorStoreIndex
import chromadb
from llama_index.legacy.vector_stores import ChromaVectorStore, ExactMatchFilter, MetadataFilters

from pipeline.llm_ident import giga_llama_llm
from llama_index.legacy.response_synthesizers import get_response_synthesizer, ResponseMode
from pipeline.prompts import qa_template, refine_template
from pipeline import questions


def get_program_name_from_file(file_name):
    """
    Получить название программы из названия файла
    :param file_name: название файла
    :return: название программы (ТОЛЬКО ЦИФРЫ)
    """
    return file_name.split('.')[0].split(' ')[1]


db = chromadb.PersistentClient(path='../VDB_new_splitter')
chroma_collection = db.get_collection(name="token_splitter")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')

service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=giga_llama_llm)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                           service_context=service_context)

llm_response_generator = get_response_synthesizer(service_context=service_context,
                                                  response_mode=ResponseMode.COMPACT,
                                                  text_qa_template=qa_template,
                                                  refine_template=refine_template)

dir = 'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/программы_для_тестов/'
files = os.listdir(dir)

if __name__ == '__main__':
    for file_name in files:
        file_output = []
        print(f'ПОИСК ПО ДОКУМЕНТУ {file_name}')
        program_name = get_program_name_from_file(file_name=file_name)
        questions_list = questions.create_question_list(file_name=file_name)
        for question in questions_list:
            text, nodes_scores, context = [], [], []
            print(f'ОБРАБОТКА ЗАПРОСА: {question}')
            base_retriever = index.as_retriever(similarity_top_k=6,
                                                filters=MetadataFilters(filters=[ExactMatchFilter(key='program_name',
                                                                                                  value=program_name)]))
            print('СТАРТ ПОИСКА РЕЛЕВАНТНЫХ ЧАНКОВ')
            retrieved_nodes = base_retriever.retrieve(question)
            for node in retrieved_nodes:
                text.append(node.text.replace('\n', ' '))
                context.append({'page_number': node.metadata['page_number'],
                                'context_lines': {'start_line': node.metadata['start_line'],
                                                  'end_line': node.metadata['end_line']}
                                })
                nodes_scores.append(node.score)
            print('ОТПРАВКА ЗАПРОСА С КОНТЕКСТОМ В МОДЕЛЬ')
            response = llm_response_generator.get_response(query_str=question, text_chunks=text)
            file_output.append({
                'question': question,
                'context': context,
                'text': text,
                'response': response,
                'nodes_score': nodes_scores
            })
        print('ЗАПИСЬ В ФАЙЛ')
        with open(f'C:/Users/ADM/OneDrive/Desktop/RAG/token_splitter/{program_name}.json', 'w', encoding="utf-8") as f:
            json.dump(file_output, f, ensure_ascii=False, indent=4)
