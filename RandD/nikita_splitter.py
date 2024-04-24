import os
import re
import json
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Literal

import httpx
from tqdm import tqdm
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate

load_dotenv()


PROMPT_TEMPLATE = """Ваша задача - составить краткое описание пункта правил из постановления правительства.

Краткое описание должно быть размером около {words} слов и {sentences} предложения.


ПУНКТ ПРАВИЛ ПОСТАНОВЛЕНИЯ ПРАВИТЕЛЬСТВА:
"{text}"


КРАТКОЕ ОПИСАНИЕ:
"""
PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=['words', 'sentences', 'text'])
START_INDEX = 'START'
SPLIT_DELIMETR = '\n' + '-' * 50 + '\n'
SUBPIECE_DELIMETR = '\n-'

SPLITS_DIR_PATH = Path('SPLITS')


def split_points(text: str) -> Dict[str, Dict[str, str]]:
    points = re.split('\n\d+\. ', text)
    text_pieces = {}
    for num, point in enumerate(points):
        subpoints = re.split('\n[a-я]{1}\)', point)
        text_subpieces = {}
        for subnum, subpoint in enumerate(subpoints):
            subpoint_chr = chr(ord('Я') + subnum)
            subkey = (subpoint_chr + ')') if subnum > 0 else START_INDEX
            text_subpieces[subkey] = subpoint.strip()
        key = (str(num) + '.') if num > 0 else START_INDEX
        text_pieces[key] = text_subpieces
    return text_pieces


def resolve_splits(pieces: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    resolve_pieces = deepcopy(pieces)
    for key, piece in resolve_pieces.items():
        for subkey, subpiece in piece.items():
            resolve_pieces[key][subkey] = re.sub(r'\s{2,}', ' ', re.sub(r'\n', ' ', subpiece)).strip()
    return resolve_pieces


def get_standard_splits(pieces: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    tmp_pieces = deepcopy(pieces)
    result = {}
    for num_piece, piece in tmp_pieces.items():
        result[num_piece] = '\n'.join([piece.pop(START_INDEX), *list(map(' '.join, piece.items()))])
    return result


def get_expanded_splits(pieces: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    tmp_pieces = deepcopy(pieces)
    result = {}
    for num, piece in tmp_pieces.items():
        if len(piece) == 1:
            result[num] = piece[START_INDEX]
        else:
            prefix = piece.pop(START_INDEX)
            for subnum, subpiece in piece.items():
                result[' '.join([num, subnum])] = prefix + subpiece
    return result


# def summary_text(text: str, num_words: int, num_sentence: int) -> str:
#     llm = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0, http_client=httpx.Client(proxies={
#         "http://": os.environ['OPENAI_PROXY'],
#         "https://": os.environ['OPENAI_PROXY'],
#     }, timeout=httpx.Timeout(60.0)))
#     chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT.partial(
#         words=str(num_words), sentences=str(num_sentence)))
#     docs = [Document(page_content=text)]
#     summarize_text = chain.run(docs)
#     return summarize_text


# def get_summary_splits(pieces: Dict[str, str], how: Literal['avg', 'max', 'min']) -> Dict[str, str]:
#     len_info = get_len_info(pieces)
#     result = {}
#     for num, piece in tqdm(pieces.items()):
#         summary_split = summary_text(piece, int(len_info['wrd'][how]), int(len_info['snt'][how]))
#         result[num] = summary_split
#     return result


def get_len_info(splits: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    len_splits_map = {
        'chr': lambda split_text: len(split_text),
        'wrd': lambda split_text: len(split_text.split()),
        'snt': lambda split_text: len(split_text.split('.')),
    }
    len_info = {}
    for key, func in len_splits_map.items():
        list_len = list(map(func, splits.values()))
        len_info[key] = {
            'list_len': list_len,
            'avg': sum(list_len) / len(list_len),
            'max': max(list_len),
            'min': min(list_len),
        }
    return len_info


def write_splits(filepath: Path, splits: Dict[str, str]) -> None:
    splits_len_info = get_len_info(splits)
    splits_info = {
        'splits': splits,
        'len_info': splits_len_info,
    }
    with open(filepath, 'w') as file:
        json.dump(splits_info, file, indent=2, ensure_ascii=False)
    print(f'Записан файл {filepath}')


if __name__ == '__main__':
    text = """
    Субсидии  предоставляются  получателям  субсидий  для  заключения  и  исполнения  кредитных
договоров (соглашений) со льготной процентной ставкой, соответствующих следующим требованиям:
а)  кредитный  договор  (соглашение)  заключен  после  дня  вступления  в  силу  постановления
Правительства  Российской  Федерации  от  2  апреля  2022  г.  N  574  "Об  утверждении  Правил
предоставления  субсидий  из  федерального  бюджета  кредитным  организациям  на  возмещение
недополученных  ими  доходов  по  кредитам,  выданным
системообразующим  организациям
топливно-энергетического  комплекса  и  организациям,  входящим  в  группу  лиц  системообразующей
организации топливно-энергетического комплекса";
б)  условия  кредитного  договора  (соглашения)  предусматривают  установление  льготной
процентной ставки на период льготного кредитования;
в)  кредитный  договор
(соглашение)  содержит  условие,  в  соответствии  с  которым
предоставленные  заемщику  средства  не  могут  быть  размещены  на  депозитах,  а  также  в  иных
финансовых  инструментах,  продажа  или  передача  которых  обеспечивает  получение  денежных  средств
(ценные бумаги, денежные обязательства, фьючерсы и опционы, прочие финансовые инструменты);
Информация  об  изменениях: Подпункт  "г"  изменен  с  22  апреля  2022  г.  -  Постановление
Правительства России от 21 апреля 2022 г. N 723
См. предыдущую редакцию
г)  кредитный  договор  (соглашение)  не  предусматривает  взимания  с  заемщика  комиссий  и
сборов,  иных  платежей,  за  исключением  платы  за  пользование  лимитом  кредитной  линии  (за
резервирование  кредитной  линии),  взимаемой  за  не  использованный  заемщиком  остаток  лимита
кредитной  линии,  комиссии  за  досрочное  погашение  в  размере  не  более  1,5  процента  годовых  от
досрочно  погашаемой  суммы  по  кредитному  договору  (соглашению),  а  также  штрафных  санкций  в
случае неисполнения заемщиком условий кредитного договора (соглашения);
д)  кредитный  договор  (соглашение)  предусматривает  получение  заемщиком  кредита  в  рублях,
размер которого рассчитывается исходя из одной четвертой выручки заемщика за 2021 год, умноженной
на  0,7,  и  не  превышает  10  млрд.  рублей,  а  для  юридических  лиц,  входящих  в  одну  группу  лиц  одной
системообразующей  организации  (включая  эту  системообразующую  организацию),  -  не  превышает  30
млрд. рублей;
е) кредитный договор (соглашение) предусматривает получение заемщиком кредита по льготной
процентной ставке в размере не более 11 процентов годовых;
ж)  кредитный  договор  (соглашение)  содержит  условие  о  запрете  на  объявление  и  выплату
(распределение  прибыли)  заемщиком  в  течение  действия  кредитного  договора
дивидендов
(соглашения),  за  исключением  случаев,  предусмотренных  отдельными  решениями  Правительства
Российской Федерации.
    """

    # READ
    # text = Path('_ПП 574.txt').read_text()
    #
    # # RESOLVE
    # raw_splits = split_points(text)
    # clear_splits = resolve_splits(raw_splits)
    # with open(SPLITS_DIR_PATH / 'source.json', 'w') as file:
    #     json.dump(clear_splits, file, indent=2, ensure_ascii=False)

    # #SPLITS
    # standard_splits = get_standard_splits(clear_splits)
    # write_splits(SPLITS_DIR_PATH / 'standard_splits.json', standard_splits)
    # expanded_splits = get_expanded_splits(clear_splits)
    # write_splits(SPLITS_DIR_PATH / 'expanded_splits.json', expanded_splits)
    #
    # avg_summary_standard_splits = get_summary_splits(standard_splits, how='avg')
    # write_splits(SPLITS_DIR_PATH / 'avg_summary_standard_splits.json', avg_summary_standard_splits)
    # max_summary_standard_splits = get_summary_splits(standard_splits, how='max')
    # write_splits(SPLITS_DIR_PATH / 'max_summary_standard_splits.json', max_summary_standard_splits)
    # min_summary_standard_splits = get_summary_splits(standard_splits, how='min')
    # write_splits(SPLITS_DIR_PATH / 'min_summary_standard_splits.json', min_summary_standard_splits)
    #
    # avg_summary_expanded_splits = get_summary_splits(expanded_splits, how='avg')
    # write_splits(SPLITS_DIR_PATH / 'avg_summary_expanded_splits.json', avg_summary_expanded_splits)
    # max_summary_expanded_splits = get_summary_splits(expanded_splits, how='max')
    # write_splits(SPLITS_DIR_PATH / 'max_summary_expanded_splits.json', max_summary_expanded_splits)
    # min_summary_expanded_splits = get_summary_splits(expanded_splits, how='min')
    # write_splits(SPLITS_DIR_PATH / 'min_summary_expanded_splits.json', min_summary_expanded_splits)






