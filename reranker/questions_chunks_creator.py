"""
Скрипт для генерации вопросов к чанку.
В текушей реализации генерируется один вопрос к чанку
"""
import json
import os

from tqdm import tqdm

from pipeline.llm_ident import gpt_llm
from pipeline.prompts import synth_questions_prompt


def create_synth_question(text: str) -> str:
    chain = synth_questions_prompt | gpt_llm
    question = chain.invoke({'text': text}).content
    return question


if __name__ == '__main__':
    dir = '/chunks_new_summarize_prompt'
    output = []
    files = os.listdir(dir)
    for file in files:
        print(f'СОЗДАНИЕ СИНТ ВОПРОСОВ ДЛЯ ФАЙЛА {file}')
        chunk_question = []
        with open(f'{dir}/{file}', 'r', encoding='utf-8') as fin:
            chunks = json.load(fin)
        for chunk in tqdm(chunks):
            try:
                question = create_synth_question(text=chunk['parent_text'])
                chunk_question.append({'chunk': chunk['text'],
                                       'question': question})
            except:
                print('ПРИ ГЕНЕРАЦИИ ВОПРОСА ПРОИЗОШЛА ОШИБКА')
        with open(f'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/reranker/paragraph_questions/{file}', 'w', encoding='utf-8') as fout:
            json.dump(chunk_question, fout, indent=4)
