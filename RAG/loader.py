import json
import re
from typing import List, Dict

from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from llama_index.legacy.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from llama_index.legacy.vector_stores import ChromaVectorStore
from llama_index.legacy.storage import StorageContext
from llama_index.legacy import ServiceContext, VectorStoreIndex
from llama_index.legacy.schema import TextNode

from pipeline.llm_interface import gigachat
from pipeline.prompts_templates import SUMMARIZE_TEXT_TMPL
from pipeline.pdf_parser import PdfMinerParser


class Splitter:
    """ Разделитель текста программы господдержки на чанки """
    #TODO: подумать, сделать ли так, чтобы Сплиттер принимал сразу список страниц в плане текста, а не объекты Document Llama-index
    def __init__(self, documents: List[Document], file_id: str):
        self.documents = documents
        self.file_id = file_id

    @staticmethod
    def _summarize(text: str) -> str:
        """
        Суммаризировать текст с помощью LLM
        :param text: исходный текст
        :return: суммаризированный текст
        """
        valid_text = text.replace('\n', ' ')
        prompt = PromptTemplate.from_template(template=SUMMARIZE_TEXT_TMPL)
        chain = prompt | gigachat
        return chain.invoke({'text': valid_text}).content

    def split(self) -> List[Dict[str, str]]:
        """
        Разделить текст на чанки по типу parent-child, где parent - пункт программы господдержки, child - суммаризированный пункт
        :param:
        :return: список чанков, где каждый чанк - словарь
        """
        chunks = []
        clean_lines, valid_lines = [], []
        for doc in self.documents:
            lines = doc.text.split('\n')
            lines.pop(0)
            for i in range(3):
                lines.pop(-1)
            for line in lines:
                if 'См. предыдущую редакцию' in line:
                    lines.remove(line)
            clean_lines.extend(lines)
        i = 0
        while i != len(clean_lines) and 'Приложение' not in clean_lines[i] and 'ПРИЛОЖЕНИЕ' not in clean_lines[i]:
            valid_lines.append(clean_lines[i])
            i += 1
        full_text = '\n'.join(valid_lines)
        paragraphs = re.split('\n\d{1,}\.{1} ', full_text)
        id = 0
        print('Создание чанков')
        for par in tqdm(paragraphs):
            parent_id = f'{self.file_id}-{id}'
            try:
                summarize_par = self._summarize(text=par)
                chunk = {
                    'text': summarize_par,
                    'parent_id': parent_id,
                    'parent_text': par,
                }
                chunks.append(chunk)
                id += 1
            except:
                pass
        return chunks


class Loader:
    """ Загрузчик документа программы господдержки в базу данных """
    #TODO: подумать, нужно ли избавиться ОТ Llama-index и перейти на Elasticsearch (да, моя хотелка)
    #TODO: подумать, нужно ли сделать загрузку других расширений кроме pdf
    def __init__(self, fpath: str):
        self.fpath = fpath
        self.embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
        self.db_conn = chromadb.PersistentClient(path='RAG/db/VDB')
        self.collection = self.db_conn.get_or_create_collection(name='main')
        self.storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=self.collection))
        self.service_context = ServiceContext.from_defaults(embed_model=self.embed_model, llm=gigachat)

    @staticmethod
    def _get_program_number(file_name: str) -> str:
        #TODO: сделать определение номера программы господдержки из текста программы, а не названия файла (задача XOPS360-2306)
        """
        Получить номер программы господдержки из документа
        :param file_name: название файла
        :return: номер программы господдержки
        """
        return file_name.split('ПП ')[1]

    @staticmethod
    def _get_program_name(program_text: str) -> str:
        #TODO: получить название/описание программы господдржки с помощью промпта по задаче Юли
        """
        Получить назване/описание программы господдержки из текста программы
        :param program_text: текст из программы господдержки, из которого определяется название/описание программы
        :return: название/описание программы господдержки
        """
        return 'F'

    def load(self):
        """
        Загрузить программу господдержки в базу данных
        """
        parents, nodes = [], []
        documents = PdfMinerParser().parse(fpath=self.fpath)
        file_name = self.fpath.split('/')[-1].split('.')[0]
        program_number = self._get_program_number(file_name=file_name)
        program_name = self._get_program_name(program_text='text_from_gos_program')
        with open('available_programs.json', 'r') as f:
            available_programs = json.load(f)
        if program_number not in available_programs['available_program_numbers']:
            chunks = Splitter(documents=documents, file_id=program_number).split()
            id = 0
            for chunk in chunks:
                parent = {'id': chunk['parent_id'],
                          'text': chunk['parent_text']}
                node_id = f'{program_number}-{id}'
                node = TextNode(text=chunk['text'], id_=node_id, metadata={'program_number': program_number,
                                                                           'program_name': program_name,
                                                                           'parent_id': chunk['parent_id']})
                id += 1
                parents.append(parent)
                nodes.append(node)
            print('Загрузка нод в векторную базу данных')
            for node in nodes:
                VectorStoreIndex(nodes=[node], storage_context=self.storage_context, service_context=self.service_context)
            print('Загрузка родителей в базу типа key-value (пока это json)')
            #TODO: подумать, надо ли внедрять key-value бд для хранения parent-чанков или JSON хватает
            try:
                with open('db/parents.json', 'r') as f:
                    all_parents = json.load(f)
                all_parents.extend(parents)
            except:
                all_parents = parents
            with open('db/parents.json', 'w') as f:
                json.dump(all_parents, f, ensure_ascii=False)
            # Сохраняем название и номер нового загруженного документа
            available_programs['available_program_numbers'].append(program_number)
            available_programs['available_program_names'].append(program_name)
            with open('available_programs.json', 'w') as f:
                json.dump(available_programs, f, ensure_ascii=False)
        else:
            print(f'Документ {file_name} уже загружен в базу данных')
