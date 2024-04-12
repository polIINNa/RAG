import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from llama_index.legacy.schema import TextNode

from pdf_parser import PdfMinerParser
from llm_ident import giga_langchain_llm


def _summarize(text: str) -> str:
    query_tmpl_str = """"
    Тебе будет дан текст из документа по программе государственной поддержки.
    Твоя задача - просуммаризировать данный текст и КРАТКО РАССКАЗАТЬ, О ЧЕМ ОН. 
    ОЧЕНЬ ВАЖНО: размер твоего ответа не должен превышать 1024 символа.

    Текст: {text}
    Суммаризация: 
    """
    text = text.replace('\n', ' ')
    prompt = PromptTemplate.from_template(template=query_tmpl_str)
    chain = prompt | giga_langchain_llm
    return chain.invoke({'text': text}).content


regex = r'\d{1,}\.{1} '
max_node_length = 1024
parser = PdfMinerParser()
path = 'C:/Users/ADM/Downloads/ПП 141.pdf'
docs = parser.parse(fpath=path)


def my_splitter(documents):
    nodes = []
    for page in documents:
        page_paragraphs = re.split(regex, page.text)
        for page_paragraph in page_paragraphs:
            metadata = {'page_number': page.metadata['page_number'],
                        'parent_node_text': page_paragraph}
            if len(page_paragraph) > max_node_length:
                paragraph_summarize = _summarize(text=page_paragraph)
                nodes.append(TextNode(text=paragraph_summarize, metadata=metadata))
            else:
                nodes.append(TextNode(text=page_paragraph, metadata=metadata))
    return nodes

#     for paragraph in paragraphs:
#         #TODO: такой поиск символа во всем тексте конечно не оптимальный, надо подумать, как сделать лучше
#         if ';' in paragraph:
#             print('СТАРТ РАЗБИЕНИЯ ПУНКЫ НА ПОДПУНКТЫ')
#             sub_paragraphs = paragraph.split(';')
#             chunks.extend(sub_paragraphs)
#         else:
#             chunks.append(paragraph)
#
# relevant_size_chunks = create_relevant_size_chunks(chunks=chunks)
# for chunk in relevant_size_chunks:
#     print(chunk, '\n\n')
# if __name__ == '__main__':
#     pass