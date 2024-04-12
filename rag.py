import json
import os

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.indices import VectorStoreIndex
import chromadb
from llama_index.legacy.vector_stores import ChromaVectorStore, ExactMatchFilter, MetadataFilters

from llm_ident import giga_llama_llm
from llama_index.legacy.response_synthesizers import get_response_synthesizer, ResponseMode
from prompts import qa_template, refine_template
import questions


def get_program_name_from_file(file_name):
    """
    Получить название программы из названия файла
    :param file_name: название файла
    :return: название программы
    """
    return file_name.split('.')[0].split(' ')[1]


db = chromadb.PersistentClient(path="VDB")
chroma_collection = db.get_collection(name="141_pp_token_splitter")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')

service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=giga_llama_llm)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                           service_context=service_context)

llm_response_generator = get_response_synthesizer(service_context=service_context,
                                                  response_mode=ResponseMode.COMPACT,
                                                  text_qa_template=qa_template,
                                                  refine_template=refine_template)

dir = 'C:/Users/ADM/Downloads/'
# files = os.listdir(dir)
files = ['ПП 141.pdf']

if __name__ == '__main__':
    for file_name in files:
        file_output = []
        print(f'ПОИСК ПО ДОКУМЕНТУ {file_name}')
        program_name = get_program_name_from_file(file_name=file_name)
        questions_list = questions.create_question_list(file_name=file_name)
        for question in questions_list:
            text, nodes_scores = [], []
            context = []
            print(f'ОБРАБОТКА ЗАПРОСА: {question}')
            base_retriever = index.as_retriever(similarity_top_k=6,
                                                filters=MetadataFilters(filters=[ExactMatchFilter(key='program_name',
                                                                                                  value=program_name)]))
            print('СТАРТ ПОИСКА РЕЛЕВАНТНЫХ ЧАНКОВ')
            retrieved_nodes = base_retriever.retrieve(question)
            for node in retrieved_nodes:
                context.append({'page_number': node.metadata['page_number'],
                                'context_lines': {'start_line': node.metadata['position_start_line'],
                                                  'end_line': node.metadata['position_end_line']}})

                text.append(node.text)
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
        with open(f'C:/Users/ADM/Downloads/{program_name}.json', 'w') as f:
            json.dump(file_output, f, ensure_ascii=False, indent=4)
