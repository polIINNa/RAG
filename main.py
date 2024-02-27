import asyncio
import logging
import sys
import os
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, Router, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.markdown import hbold

from RAG.rag import RAG

# Bot token can be obtained via https://t.me/BotFather
load_dotenv()
TOKEN = os.environ['TELEGRAM_API_TOKEN']

# All handlers should be attached to the Router (or Dispatcher)
dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(f"Привет <b>{message.from_user.first_name}</b>!\n"
                         f"Это бот, который отвечает на вопросы по документам по господдержке. Напишите вопрос - будем искать на него ответ)")


@dp.message()
async def main_handler(message: Message, bot: Bot) -> None:
    """
    By default, message handler will handle all message types (like a text, photo, sticker etc.)
    """
    # try:
    await message.answer(f'Запускаю обработку вопроса: <b>{message.text}</b>')
    if message.text is not None and message.from_user is not None:
        rag = RAG()
        response = await rag.process(query=message.text, bot=bot, user_id=message.from_user.id)
        await message.answer(f'<b>Ответ</b>:\n{response}')
    else:
        await message.answer('Не могу обработать такой формат запроса')
    # except Exception as error:
    #     await message.answer("Возникла ошибка... Кажется, Вам придется идти читать документ самим :(")


async def main() -> None:
    # Initialize Bot instance with a default parse mode which will be passed to all API calls
    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
