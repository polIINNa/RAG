import json
import os
from typing import List

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
    dir = 'summarize_data/chunks'
    files = os.listdir(dir)
    for file in files:
        print(f'ОБРАБОТКА ФАЙЛА {file}')
        output = []
        with open(f'{dir}/{file}', encoding='utf-8') as f:
            file_chunks = json.load(f)
        for chunk in file_chunks:
            try:
                questions = create_synth_questions(text=chunk['text'])
                output.append({'chunk': chunk['text'],
                               'questions': questions})
            except:
                print('ПРИ ГЕНЕРАЦИИ/ВАЛИДАЦИИ ВОПРОСОВ ПРОИЗОШЛА ОШИБКА')
                output.append({'chunk': chunk['text'],
                               'questions': None})
        print('ЗАПИСЬ ВОПРОСОВ В ФАЙЛ')
        with open(f'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/chunks_questions/{file}', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4)


