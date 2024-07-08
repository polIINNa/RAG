import json

from fastapi import FastAPI, status

from RAG.rag import RAG
from fast_api.message import Message

with open('/Users/21109090/Desktop/RAG_gospodderzka/RAG/available_programs.json', 'r') as f:
    available_programs = json.load(f)

app = FastAPI(description=f"Это сервис, который отвечает на вопросы по документам по господдержке.\n"
                          f"Вот список доступных документов: {available_programs['available_program_numbers']}.\n"
                          f"<b>!ВАЖНО!</b>: на текущий момент я могу отвечать на вопрос, "
                          f"только если в нем есть номер постановления, например: "
                          f"<i>'В чем суть постановления 295?'</i>")


@app.get("/api/v1/healthcheck", status_code=status.HTTP_200_OK, response_model=Message,
         summary='Проверка работоспособности сервиса')
async def healthcheck():
    return Message(body="OKResponce")


@app.post("/api/v1/question", response_model=Message, status_code=status.HTTP_200_OK,
          summary='Вопрос по документам по господдержке')
async def parse_question(message: Message):
    rag_gs = RAG()
    gs_program = rag_gs.get_gs_program(query=message.body)
    if gs_program is None:
        return Message(body='Не удалось определить программу господдержики, в которой надо искать ответ на вопрос')
    else:
        if gs_program[1] is False:
            return Message(body=f'Определенная из вопроса программа: <b>{gs_program[0]}</b>. К сожалению, данной программы нет в базе знаний, обратитесь к команде проекта.')
        else:
            addition = f"Постановление, в котором происходил поиск ответа на вопрос: {gs_program[0]}."
            rag_answer = rag_gs.rag(query=message.text, gs_program_name=gs_program[0])
            return Message(body=f'<b>Ответ на вопрос</b>\n{rag_answer}. {addition}')
