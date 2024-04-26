import json
import os

import pandas as pd


dir = 'C:/Users/ADM/OneDrive/Desktop/RAG/summarize_query_rewriting/no_doc_info_rephrase/eval_results'
files = os.listdir(dir)

program_names, origin_questions, rewrite_questions, contexts, nodes_scoress, llm_responses, \
len_gold_contexts, numb_match_liness, context_recalls, avg_nodes_scoress, \
answer_correctnesss, context_precisions = [], [], [], [], [], [], [], [], [], [], [], []

if __name__ == '__main__':
    for file_name in files:
        print(f'ОБРАБОТКА РЕЗУЛЬТАТОВ ФАЙЛА {file_name}')
        with open(f'{dir}/{file_name}', 'r', encoding='utf-8') as f:
            data = json.load(f)
        program_name = file_name.split('.')[0]
        for res in data:
            program_names.append(program_name)
            origin_questions.append(res['origin question'])
            rewrite_questions.append(res['rewrite question'])
            contexts.append(res['text'])
            llm_responses.append(res['llm_response'])
            answer_correctnesss.append(res['answer_correctness_rewrite'])
            nodes_scoress.append(res['nodes_score'])
            avg_nodes_scoress.append(sum(res['nodes_score'])/len(res['nodes_score']))

            if res['context_eval'] is not None:
                len_gold_contexts.append(res['context_eval']['len_gold_context'])
                numb_match_liness.append(res['context_eval']['numb_match_lines'])
                context_recalls.append(res['context_eval']['context_recall'])
                context_precisions.append(res['context_eval']['context_precision'])
            else:
                len_gold_contexts.append('null')
                numb_match_liness.append('null')
                context_recalls.append('null')
                context_precisions.append('null')

    res2write = {'Номмер программы': program_names,
                 'Оригинальный вопрос': origin_questions,
                 'Перефразированный вопрос': rewrite_questions,
                 'Контекст': contexts,
                 'Значения косинусной близости нод': nodes_scoress,
                 'Среднее значение скора по нодам': avg_nodes_scoress,
                 'Ответ модели': llm_responses,
                 'Число предложений в голд контексте': len_gold_contexts,
                 'Число мэтча предложений и голд контекста': numb_match_liness,
                 'Recall': context_recalls,
                 'Precision': context_precisions,
                 'Answer correctness (rewrite)': answer_correctnesss}

    df = pd.DataFrame(res2write)
    df.to_excel('C:/Users/ADM/OneDrive/Desktop/RAG/summarize_query_rewriting_no_doc_info_rephrase.xlsx')



