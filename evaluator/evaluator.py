import json

from datasets import Dataset
from ragas.metrics import answer_correctness
from ragas import evaluate

from pipeline.llm_ident import gpt_llm
from langchain.embeddings import HuggingFaceEmbeddings


EMBED_MODEL = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')


def get_mapped_questions(program_name):
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


def get_file_context_gold_markup(gold_markup, file_name):
    """
    Получить из всей разметки разметку по нужному вопросу
    :param gold_markup: вся разметка
    :param file_name: имя файла, по которому нужна разметка
    :return: разметка по файлу
    """
    for i in range(len(gold_markup)):
        if gold_markup[i]['file_name'].split('.')[0].split(' ')[1] == file_name:
            return gold_markup[i]['markup']


def get_file_answer_gold_markup(file_name, answer_gold_markup):
    for file in answer_gold_markup:
        if file['file_name'].split('.')[0].split(' ')[1] == file_name:
            return file['answer_gold_markup']


def get_gold_context(question, mapped_questions, file_gold_markup):
    """
    Получить голд контекст по вопросу
    """
    mapped_question = mapped_questions[question]
    for i in range(len(file_gold_markup)):
        if file_gold_markup[i]['question'] == mapped_question:
            return file_gold_markup[i]['context']


def get_gold_answer(question, mapped_questions, file_gold_markup):
    """
    Получить голд ответ по вопросу
    """
    mapped_question = mapped_questions[question]
    for data in file_gold_markup:
        if data['question'] == mapped_question:
            return data['answer']


def get_len_gold_context(gold_context):
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


def get_len_context(context):
    """
    Получить размер контекста (всех чанков
    """
    len_context = 0
    for chunk in context:
        len_context += chunk['context_lines']['end_line'] - chunk['context_lines']['start_line'] + 1
    return len_context


def eval_context(context, gold_context):
    """
    Получить оценку по одному вопросу
    :param context: контекст по вопросу
    :param gold_context: голд контекст по вопросу
    :return: метрик recall и precision
    """
    numb_match_lines = 0
    len_gold_context = get_len_gold_context(gold_context=gold_context)
    len_context = get_len_context(context=context)
    if len_gold_context != 0:
        for page in gold_context:
            for chunnk in context:
                if chunnk['page_number'] == page['page_number']:
                    chunk_lines = set([line for line in range(chunnk['context_lines']['start_line'], chunnk['context_lines']['end_line']+1)])
                    for box in page['context_lines']:
                        box_lines = set([line for line in range(box['start_line'], box['end_line']+1)])
                        numb_match_lines += len(box_lines.intersection(chunk_lines))
        recall = numb_match_lines/len_gold_context
        precision = numb_match_lines/len_context
        context_eval = {'len_gold_context': len_gold_context,
                        'len_context': len_context,
                        'numb_match_lines': numb_match_lines,
                        'recall': recall,
                        'precision': precision}
        return context_eval


def eval_answer(gold_answer, rag_answer, question):
    data_samples = {
        'question': [question],
        'answer': [rag_answer],
        'ground_truth': [gold_answer]
    }
    dataset = Dataset.from_dict(data_samples)
    if gold_answer != '-':
        answer_correctness.weights = [0.9, 0.1]
        score = evaluate(dataset, metrics=[answer_correctness],
                         embeddings=EMBED_MODEL, llm=gpt_llm)
        return score['answer_correctness']
    else:
        return None


with open('C:/Users/ADM/OneDrive/Desktop/RAG/gold_markup.json', 'r', encoding='utf-8') as f:
    context_gold_markup = json.load(f)
with open('C:/Users/ADM/OneDrive/Desktop/RAG/answer_gold_markup.json', 'r', encoding='utf-8') as f:
    answer_gold_markup = json.load(f)

dir = 'C:/Users/ADM/OneDrive/Desktop/RAG/token_splitter'
files_res = ['1598.json', '2221.json', '574.json']
if __name__ == '__main__':
    for file in files_res:
        file_eval = []
        print(f'РАСЧЕТ МЕТРИК ДЛЯ ФАЙЛА {file}')
        with open(f'{dir}/{file}', 'r', encoding='utf-8') as f:
            file_res = json.load(f)
        file_context_gold_markup = get_file_context_gold_markup(file_name=file.split('.')[0],
                                                                gold_markup=context_gold_markup)
        file_answer_gold_markup = get_file_answer_gold_markup(file_name=file.split('.')[0],
                                                              answer_gold_markup=answer_gold_markup)
        for res in file_res:
            mapped_questions = get_mapped_questions(program_name=file.split('.')[0])

            gold_context = get_gold_context(question=res['question'], mapped_questions=mapped_questions,
                                            file_gold_markup=file_context_gold_markup)
            gold_answer = get_gold_answer(question=res['question'], mapped_questions=mapped_questions,
                                          file_gold_markup=file_answer_gold_markup)

            answer_eval = eval_answer(gold_answer=gold_answer, rag_answer=res['response'], question=res['question'])
            context_eval = eval_context(context=res['context'], gold_context=gold_context)
            file_eval.append({'question': res['question'],
                              'text': res['text'],
                              'context_eval': context_eval,
                              'llm_response': res['response'],
                              'answer_correctness': answer_eval,
                              'nodes_score': res['nodes_score']})
        with open(f'C:/Users/ADM/OneDrive/Desktop/RAG/token_splitter/eval_results/{file}', 'w', encoding='utf-8') as f:
            json.dump(file_eval, f, ensure_ascii=False, indent=4)

