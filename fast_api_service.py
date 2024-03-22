from fastapi import FastAPI, status, HTTPException

from RAG.rag import RAG
from fast_api.message import Message
from programs import AVAILABLE_PROGRAMS

app = FastAPI(description=f"Это сервис, который отвечает на вопросы по документам по господдержке.\n"
                          f"Вот список доступных документов: {AVAILABLE_PROGRAMS}.\n"
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
    rag = RAG()
    program_number = rag.get_program_number(query=message.body)
    if program_number == '-1':
        raise HTTPException(
            status_code=status.HTTP_200_OK,
            detail="Пока реализация такова, что в вопросе должен присутствовать номер документа. "
                   "Введите вопрос с номером постановления"
        )
    elif program_number not in AVAILABLE_PROGRAMS:
        raise HTTPException(
            status_code=status.HTTP_200_OK,
            detail=f"Таких постановлений нет в базе. Вот список доступных: {AVAILABLE_PROGRAMS}"
        )
    else:
        caption = f"Постановления, в которых буду искать ответ на вопрос: {program_number}."
        response = await rag.process(query=message.body)
        return Message(body=f"{caption} {response}")
