from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from RAG.rag import RAG
from fast_api.message import Message
from programs import AVAILABLE_PROGRAMS

app = FastAPI()


@app.get("/api/info")
async def get_info():
    answer = Message(
        body=f"Это сервис, который отвечает на вопросы по документам по господдержке. "
             f"Вот список доступных документов: {AVAILABLE_PROGRAMS}."
             f"!ВАЖНО!: на текущий момент я могу отвечать на вопрос, "
             f"только если в нем есть номер постановления, например: 'В чем суть постановления 295?'"
    )
    return answer


@app.post("/api/question/")
async def parse_question(message: Message):
    if message.body != '':
        rag = RAG(query=message.body)
        program_number = rag.get_program_number(query=message.body)
        if program_number == '-1':
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": f"Пока реализация такова, что в вопросе должен присутствовать номер документа. "
                                    f"Введите вопрос с номером постановления"}
            )
        elif program_number not in AVAILABLE_PROGRAMS:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"Таких постановлений нет в базе. Вот список доступных: {AVAILABLE_PROGRAMS}"}
            )
        else:
            message = "Постановления, в которых буду искать ответ на вопрос: {program_number}."
            response = await rag.process()
            return Message(
                body=f"{message} {response}"
            )
    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "Вопрос должен быть текстом"}
        )
