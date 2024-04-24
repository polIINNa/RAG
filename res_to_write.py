import json
import os

import pandas as pd


dir = 'C:/Users/ADM/OneDrive/Desktop/RAG/summarize_splitter/eval_results'
files = os.listdir(dir)

program_names, questions, contexts, nodes_scoress, llm_responses, \
len_gold_contexts, numb_match_liness, context_recalls, avg_nodes_scoress, \
answer_correctnesss = [], [], [], [], [], [], [], [], [], []

if __name__ == '__main__':
    for file_name in files:
        print(f'ОБРАБОТКА РЕЗУЛЬТАТОВ ФАЙЛА {file_name}')
        with open(f'{dir}/{file_name}', 'r', encoding='utf-8') as f:
            data = json.load(f)
        program_name = file_name.split('.')[0]
        for res in data:
            program_names.append(program_name)
            questions.append(res['question'])
            contexts.append(res['text'])
            llm_responses.append(res['llm_response'])
            answer_correctnesss.append(res['answer_correctness'])
            nodes_scoress.append(res['nodes_score'])
            avg_nodes_scoress.append(sum(res['nodes_score'])/len(res['nodes_score']))

            if res['context_eval'] is not None:
                len_gold_contexts.append(res['context_eval']['len_gold_context'])
                numb_match_liness.append(res['context_eval']['numb_match_lines'])
                context_recalls.append(res['context_eval']['context_recall'])
            else:
                len_gold_contexts.append('null')
                numb_match_liness.append('null')
                context_recalls.append('null')

    res2write = {'Номмер программы': program_names,
                 'Вопрос': questions,
                 'Контекст': contexts,
                 'Значения косинусной близости нод': nodes_scoress,
                 'Среднее значение скора по нодам': avg_nodes_scoress,
                 'Ответ модели': llm_responses,
                 'Число предложений в голд контексте': len_gold_contexts,
                 'Число мэтча предложений и голд контекста': numb_match_liness,
                 'Recall': context_recalls,
                 'Answer correctness': answer_correctnesss}

    df = pd.DataFrame(res2write)
    df.to_excel('C:/Users/ADM/OneDrive/Desktop/RAG/summarize_splitter.xlsx')



