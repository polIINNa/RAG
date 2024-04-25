import re

from langchain_core.prompts import PromptTemplate

from pipeline.llm_ident import giga_langchain_llm_soft


POINT_REGEX = r'\d{1,}\.{1} '
SUBPOINT_REGEX = r'\n[a-я]{1}\)'
MAX_NODE_LENGHT = 1024


def _summarize(text):
    query_tmpl_str = """"
    Тебе будет дан текст из документа по программе государственной поддержки.
    Твоя задача - просуммаризировать данный текст и КРАТКО РАССКАЗАТЬ, О ЧЕМ ОН. 
    ОЧЕНЬ ВАЖНО: размер твоего ответа не должен превышать 1024 символа.

    Текст: {text}
    Суммаризация: 
    """
    text = text.replace('\n', ' ')
    prompt = PromptTemplate.from_template(template=query_tmpl_str)
    chain = prompt | giga_langchain_llm_soft
    return chain.invoke({'text': text}).content


def split(documents):
    chunks = []
    for page in documents:
        points = re.split(POINT_REGEX, page.text)
        for idx, point in enumerate(points):
            point_id = f'{page.metadata["file_name"]}-{page.metadata["page_number"]}-{idx}'
            if len(point) > MAX_NODE_LENGHT:
                summarize_point = _summarize(text=point)
                chunk = {
                    'text': summarize_point,
                    'page_number': page.metadata["page_number"],
                    'parent_id': point_id,
                    'parent_text': point
                }
            else:
                chunk = {
                    'text': point,
                    'page_number': page.metadata["page_number"],
                    'parent_id': point_id,
                    'parent_text': point
                }
            chunks.append(chunk)
    return chunks
