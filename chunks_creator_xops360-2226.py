
import os
import json
import re

from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

from pipeline.llm_ident import giga_langchain_llm_strict
from pipeline.pdf_parser import PdfMinerParser


def _summarize(text):
    query_tmpl_str = """"
        Тебе будет дан текст из документа по программе государственной поддержки.
        Твоя задача - сократить предоставленный текст, убрать из него всю ненужную информацию.
        ОЧЕНЬ ВАЖНО: размер твоего ответа не должен превышать 1024 символа.

        Текст: {text}
        Суммаризация:
        """
    text = text.replace('\n', ' ')
    prompt = PromptTemplate.from_template(template=query_tmpl_str)
    chain = prompt | giga_langchain_llm_strict
    return chain.invoke({'text': text}).content


if __name__ == '__main__':
    dir = 'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/программы_для_тестов'
    files = os.listdir(path=dir)
    # files = ['ПП 1598.pdf']
    for file in files:
        print(f'СОЗДАНИЕ ЧАНКОВ ДЛЯ ФАЙЛА {file}')
        docs = PdfMinerParser().parse(fpath=f'{dir}/{file}')
        clean_lines, valid_lines = [], []
        for doc in docs:
            lines = doc.text.split('\n')
            # Удаление мусорной строчки наверху страницы
            lines.pop(0)
            # Удаление мусорной строчки внизу страницы (3 последних элемента)
            for i in range(3):
                lines.pop(-1)
            # Удаление "См. предыдущую редакцию"
            for line in lines:
                if 'См. предыдущую редакцию' in line:
                    lines.remove(line)
            clean_lines.extend(lines)
        # Удаление приложений
        i = 0
        line = clean_lines[i]
        while i != len(clean_lines) and 'Приложение' not in clean_lines[i] and 'ПРИЛОЖЕНИЕ' not in clean_lines[i]:
            valid_lines.append(clean_lines[i])
            i += 1
        full_text = '\n'.join(valid_lines)
        paragraphs = re.split('\n\d{1,}\.{1} ', full_text)
        # print(len(paragraphs))
        # for p in paragraphs:
        #     print(p, '\n')
        chunks = []
        id = 0
        file_name = file.split(".")[0]
        for par in tqdm(paragraphs):
            parent_id = f'{file_name}-{id}'
            summarize_par = _summarize(text=par)
            chunk = {
                'text': summarize_par,
                'parent_id': parent_id,
                'parent_text': par
            }
            chunks.append(chunk)
            id += 1
        with open(f'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/chunks_new_paragraph_splitter/remove_application/{file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4)
