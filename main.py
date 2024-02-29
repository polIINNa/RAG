import os
import sys
import asyncio
import logging

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from RAG.rag import RAG
from programs import AVAILABLE_PROGRAMS

load_dotenv()
TOKEN = os.getenv('TELEGRAM_API_TOKEN_TEST')

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    Орбаботать команду /start
    """
    await message.answer(f"Привет, <b>{message.from_user.first_name}</b>!\n"
                         f"Это бот, который отвечает на вопросы по документам по господдержке\n"
                         f"Вот список доступных документов: {AVAILABLE_PROGRAMS}\n"
                         f"<b>!ВАЖНО!</b>: на текущий момент я могу отвечать на вопрос, только если в нем есть номер постановления, например: <i>'В чем суть постановления 295?'</i>")


@dp.message()
async def main_handler(message: Message) -> None:
    await message.answer(f'Запускаю обработку вопроса: <b>{message.text}</b>')
    if message.text is not None and message.from_user is not None:
        rag = RAG(query=message.text)
        program_number = rag.get_program_number(query=message.text)
        if program_number == '-1':
            await message.answer('Пока реализация такова, что в вопросе должен присутствовать номер документа. Пожалуйста, введите вопрос с номером постановления')
        elif program_number not in AVAILABLE_PROGRAMS:
            await message.answer(f'Ой-ой, кажется, таких постановлений нет в моей базе :(\nВот список доступных: {AVAILABLE_PROGRAMS}')
        else:
            await message.answer(f'Постановления, в которых буду искать ответ на вопрос: <b>{program_number}</b>')
            response = await rag.process()
            await message.answer(f'<b>Ответ</b>:\n{response}')
    else:
        await message.answer('Вопрос должен быть текстом')


async def main() -> None:
    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
