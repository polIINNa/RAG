import json
import os
from typing import List

from pdf_parser import PdfMinerParser
import summarize_splitter
from pipeline.llm_ident import gpt_llm
from pipeline.prompts import synth_questions_prompt


def create_synth_questions(text: str) -> List[str]:
    chain = synth_questions_prompt | gpt_llm
    response = chain.invoke({'text': text}).content
    raw_qs = response.split('\n')
    questions = []
    for raw_q in raw_qs:
        questions.append(raw_q.split(': ')[1])
    return questions


if __name__ == '__main__':
    output = []
    numb_questions = 0
    with open('chunks.json', 'r') as f:
        data = json.load(f)
    for file_data in data:
        print(f'ОБРАБОТКА ФАЙЛА {file_data["file_name"]}')
        file_questions = []
        for chunk_data in file_data['chunks']:
            try:
                chunk_questions = create_synth_questions(text=chunk_data['text'])
                numb_questions += len(chunk_questions)
                file_questions.append({'chunk': chunk_data['text'],
                                       'questions': chunk_questions})
                print(chunk_data['text'].replace('\n', ' '))
                print(chunk_questions)
                print('\n')
            except:
                print('ПРИ ГЕНЕРАЦИИ/ВАЛИДАЦИИ ВОПРОСОВ ПРОИЗОШЛА ОШИБКА')
                file_questions.append({'chunk': chunk_data['text'],
                                       'questions': 'null'})
        print('ВОПРОСЫ СГЕНЕРИРОВАНЫ')
        output.append({'file_name': file_data['file_name'],
                       'data': file_questions})
    print(numb_questions)
    with open('chunks_questions.json', 'w') as f:
        json.dump(output, f, indent=4)


