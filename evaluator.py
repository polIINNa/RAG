import os
from typing import List, Dict
import json


def get_mapped_questions(program_name: str) -> Dict:
    """
    Получить маппинг задаваемых вопрос на те, по которым сделана разметка
    :param program_name:
    :return:
    """
    program_name = 'ПП '+program_name
    mapped_questions = {
        f'Покажи основные условия программы ПП (постановление правительства) {program_name}': 'Основные условия программы ПП?',
        f'В чем суть программы {program_name}?': 'В чем суть программы ПП?',
        f'На что направлена программа {program_name}?': 'На что направлена программа ПП?',
        f'Какие компании могут претендовать на льготное финансирование в соответствии с программой {program_name}?': 'Какие компании могут претендовать на льготное финансирование в соответствие с программой?',
        f'Как высчитывается льготная ставка в рамках программы {program_name}?': 'Как высчитывается льготная ставка в рамках программы?',
        f'Какая ставка применяется по программе {program_name}?': 'Какая ставка применяется по программе?',
        f'Какая субсидия предоставляется банку в рамках программы {program_name}?': 'Какая субсидия предоставляется банку в рамках программы?',
        f'Какой срок кредитования по программе {program_name}?': 'Какой срок кредитования по программе?',
        f'Как долго может действовать льготная ставка по программе {program_name}?': 'Как долго может действовать льготная ставка по программе?',
        f'Какой порядок оформления документации в рамках программы {program_name}?': 'Какой порядок оформления документации по программе?',
        f'Кто является основными субъектами льготной программы {program_name}?': 'Кто является основными субъектами льготной программы?',
        f'Какие есть ограничения по госпрограмме {program_name}?': 'Какие есть ограничения по госпрограмме?',
        f'Когда вступила в силу программа {program_name}?': 'Когда вступила в силу программа?'
  }
    return mapped_questions


def get_file_gold_markup(gold_markup, file_name: str) -> List:
    """
    Получить из всей разметки разметку по нужному вопросу
    :param gold_markup: вся разметка
    :param file_name: имя файла, по которому нужна разметка
    :return: разметка по файлу
    """
    for i in range(len(gold_markup)):
        if gold_markup[i]['file_name'].split('.')[0].split(' ')[1] == file_name:
            return gold_markup[i]['markup']


def get_gold_context(question: str, mapped_questions: Dict, file_gold_markup: List[Dict]) -> List:
    """
    Получить голд контекст по вопросу
    """
    mapped_question = mapped_questions[question]
    for i in range(len(file_gold_markup)):
        if file_gold_markup[i]['question'] == mapped_question:
            return file_gold_markup[i]['context']


def get_len_gold_context(gold_context) -> int:
    """
    Получить размер голд контекста
    :param gold_context: голд контекст по вопросу
    :return: размер голд контекста
    """
    len_gold_context = 0
    for page in gold_context:
        for box in page['context_lines']:
            len_gold_context += box['end_line']-box['start_line']+1
    return len_gold_context


def get_eval(context, gold_context) -> Dict | None:
    """
    Получить оценку по одному вопросу
    :param context: контекст по вопросу
    :param gold_context: голд контекст по вопросу
    :return:
    """
    numb_match_lines = 0
    len_gold_context = get_len_gold_context(gold_context=gold_context)
    if len_gold_context != 0:
        for page in gold_context:
            for node in context:
                if node['page_number'] == page['page_number']:
                    node_lines = set([line for line in range(node['context_lines']['start_line'], node['context_lines']['end_line']+1)])
                    for box in page['context_lines']:
                        box_lines = set([line for line in range(box['start_line'], box['end_line']+1)])
                        numb_match_lines += len(box_lines.intersection(node_lines))
        context_eval = {'len_gold_context': len_gold_context,
                        'numb_match_lines': numb_match_lines,
                        'context_recall': numb_match_lines/len_gold_context}
        return context_eval


with open('/Users/21109090/Desktop/госпрограмма/to_eval/gold_markup.json', 'r') as f:
    gold_markup = json.load(f)

dir = '/Users/21109090/Desktop/госпрограмма/to_eval/rag_results'
files_res = os.listdir(dir)

for file in files_res:
    file_eval = []
    with open(f'{dir}/{file}', 'r') as f:
        file_res = json.load(f)
    file_gold_markup = get_file_gold_markup(file_name=file.split('.')[0], gold_markup=gold_markup)
    for res in file_res:
        mapped_questions = get_mapped_questions(program_name=file.split('.')[0])
        gold_context = get_gold_context(question=res['question'], mapped_questions=mapped_questions,
                                        file_gold_markup=file_gold_markup)
        context_eval = get_eval(context=res['context'], gold_context=gold_context)
        file_eval.append({'question': res['question'],
                          'text': res['text'],
                          'context_eval': context_eval,
                          'llm_response': res['response']})
    with open(f'/Users/21109090/Desktop/госпрограмма/to_eval/eval_results/{file}', 'w') as f:
        json.dump(file_eval, f, ensure_ascii=False, indent=4)




