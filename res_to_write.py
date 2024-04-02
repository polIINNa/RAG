import json
import os

import pandas as pd


dir = '/Users/21109090/Desktop/госпрограмма/to_eval/eval_results'
files = os.listdir(dir)

program_names, questions, contexts, llm_responses, len_gold_contexts, numb_match_liness, context_recalls = [], [], [], [], [], [], []
for file_name in files:
    print(f'ОБРАБОТКА РЕЗУЛЬТАТОВ ФАЙЛА {file_name}')
    with open(f'{dir}/{file_name}', 'r') as f:
        data = json.load(f)
    program_name = file_name.split('.')[0]
    for res in data:
        program_names.append(program_name)
        questions.append(res['question'])
        contexts.append(res['text'])
        llm_responses.append((res['llm_response']))
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
             'Ответ модели': llm_responses,
             'Число предложений в голд контексте': len_gold_contexts,
             'Число мэтча предложений и голд контекста': numb_match_liness,
             'Recall': context_recalls}

df = pd.DataFrame(res2write)
df.to_excel('/Users/21109090/Desktop/госпрограмма/to_eval/eval_results_table.xlsx')



