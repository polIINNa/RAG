from langchain_core.prompts import PromptTemplate as langchain_prompt_tmpl
from llama_index.legacy.prompts import PromptTemplate as llama_index_prompt_tmpl


QA_TEMPLATE_TEXT = """
Ты юрист и отлично разбираешься в документах, которые содержат информацию о программах государственной поддежрки компаний.
Тебе будет дан вопрос по программе и контекст, в котором содержится ответ на вопрос.

Твоя задача:
Основываясь ТОЛЬКО на данном контексте, предоставь ответ на вопрос.
Отвечай СТРОГО ПО ДАННОМУ ВОПРОСУ. 
Если в контексте есть ифнормация, которая не относится к вопросу - ИГНОРИРУЙ ее.

Контекст: {context_str}

Вопрос: {query_str}

Ответ:

"""

REFINE_PROMPT_TEMPLATE_TEXT = """
Исходный вопрос: {query_str}
Текущий ответ модели: {existing_answer}
У тебя есть возможность сделать текущий ответ более точным за счет дополнительного контекста.

Дополнительный контекст:
------------
{context_msg}
------------
Используя дополнительный контекст, уточни текущий ответ, чтобы он лучше отвечал на вопрос.
Если дополнтительный контекст не дает новой полезной информации - верни текущий ответ модели

Уточненный ответ: 
"""

PROGRAM_NAME_PROMPT_TMPL_STR = """
Тебе будет дан вопрос, который задается по программе Постановления Правительства (ПП).
В ответ напиши НОМЕР ПОСТАНОВЛЕНИЯ, в котором надо искать ответ на заданный вопрос.

ПРИМЕРЫ
Вопрос: Какие основные условия программы ПП 000?
Ответ: 000

Вопрос: Какие льготы даются по 295 ПП?
Ответ: 295

Вопрос: По программе 666 какие компании могут получить льготы?
Ответ: 666

ЕСЛИ НЕ ПОЛУЧАЕТСЯ ИЗВЛЕЧЬ НОМЕР из постановления - в ответе верни -1

ЗАПРОС
Вопрос: {query_str}
Ответ:
"""


qa_template = llama_index_prompt_tmpl(template=QA_TEMPLATE_TEXT, prompt_type='text_qa')
refine_template = llama_index_prompt_tmpl(template=REFINE_PROMPT_TEMPLATE_TEXT, prompt_type='refine')
program_name_template = langchain_prompt_tmpl.from_template(PROGRAM_NAME_PROMPT_TMPL_STR)
langchain_qa_template = langchain_prompt_tmpl.from_template(QA_TEMPLATE_TEXT)

