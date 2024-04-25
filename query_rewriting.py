
from pipeline import prompts, questions, llm_ident


def rewrite_question(question: str) -> str:
    chain = prompts.query_rewriting_prompt_tmpl | llm_ident.giga_langchain_llm_strict
    response = chain.invoke({'question': question}).content
    return response


if __name__ == '__main__':
    file_name = '1598.json'
    questions = questions.create_question_list(file_name=file_name)
    for q in questions:
        print(f'ОРИГИНАЛЬНЫЙ ВОПРОС: {q}')
        print(f'ПЕРЕФОРМУЛИРОВАННЫЙ ВОПРОС: {rewrite_question(question=q)}' '\n')
