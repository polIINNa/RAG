from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from RAG.rag import RAG
from fast_api.message import Message
from programs import AVAILABLE_PROGRAMS
from fastapi.encoders import jsonable_encoder

app = FastAPI(prefix="/api", description=f"Это сервис, который отвечает на вопросы по документам по господдержке. "
                                         f"Вот список доступных документов: {AVAILABLE_PROGRAMS}."
                                         f"!ВАЖНО!: на текущий момент я могу отвечать на вопрос, "
                                         f"только если в нем есть номер постановления, например: 'В чем суть постановления 295?'")


@app.get("/healthcheck", status_code=status.HTTP_200_OK, response_model=Message, summary='Проверка работоспособности сервиса')
async def healthcheck():
    return Message(body="Сервис работает")


@app.post("/question", response_model=Message, summary='Вопрос по документам по господдержке')
async def parse_question(message: Message):
    rag = RAG(query=message.body)
    program_number = rag.get_program_number(query=message.body)
    if program_number == '-1':
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=jsonable_encoder(Message(body="Пока реализация такова, "
                                                  "что в вопросе должен присутствовать номер документа. "
                                                  "Введите вопрос с номером постановления"))
        )
    elif program_number not in AVAILABLE_PROGRAMS:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=jsonable_encoder(Message(body=f"Таких постановлений нет в базе. "
                                                  f"Вот список доступных: {AVAILABLE_PROGRAMS}"))
        )
    else:
        message = f"Постановления, в которых буду искать ответ на вопрос: {program_number}."
        response = await rag.process()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder(Message(body=f"{message} {response}"))
        )
