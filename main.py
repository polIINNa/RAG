import os
import sys
import json
import asyncio
import logging

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from RAG.rag import RAG

load_dotenv()
TOKEN = os.getenv('TELEGRAM_API_TOKEN')

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    with open('/Users/21109090/Desktop/RAG_gospodderzka/RAG/available_programs.json', 'r') as f:
        available_programs = json.load(f)
    await message.answer(f"Привет, <b>{message.from_user.first_name}</b>!\n"
                         f"Это бот, который отвечает на вопросы по документам по господдержке\n"
                         f"Вот список доступных документов: {available_programs['available_program_numbers']}\n"
                         f"<b>!ВАЖНО!</b>: на текущий момент я могу отвечать на вопрос, только если в нем есть номер постановления, например: <i>'В чем суть постановления 666?'</i>")


@dp.message()
async def rag_handler(message: Message) -> None:
    answer = await message.answer('Запускаю обработку вопроса')
    if message.text is not None and message.from_user is not None:
        rag_gs = RAG()
        await answer.delete()
        answer = await message.answer('Определяю подходящую программы господдержки')
        gs_program = rag_gs.get_gs_program(query=message.text)
        await answer.delete()
        if gs_program is None:
            await message.answer('Не удалось определить программу господдержики, в которой надо искать ответ на вопрос')
        else:
            if gs_program[1] is False:
                await message.answer(f'Определенная из вопроса программа: <b>{gs_program[0]}</b>. К сожалению, данной программы нет в базе знаний, обратитесь к команде проекта.')
            else:
                answer = await message.answer(f'Определенная из вопроса программа: <b>{gs_program[0]}</b>. Поиск будет вестись в ней')
                gs_program_name = gs_program[0]
                rag_answer = rag_gs.rag(query=message.text, gs_program_name=gs_program_name)
                await answer.delete()
                await message.answer(f'<b>Ответ на вопрос</b>\n{rag_answer}')


async def main() -> None:
    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
