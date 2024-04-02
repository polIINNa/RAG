import os

from dotenv import load_dotenv
from llama_index.legacy.llms import LangChainLLM
from langchain_community.chat_models import GigaChat

load_dotenv()

base_url = 'https://gigachat.devices.sberbank.ru/api/v1'
auth_url = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'
scope = os.getenv('GIGA_SCOPE')
credentials = os.getenv('GIGA_CREDENTIALS')

giga_langchain_llm = GigaChat(base_url=base_url, auth_url=auth_url, scope=scope, credentials=credentials,
                              verify_ssl_certs=False, model=os.getenv('GIGA_MODEL'), profanity=False, temperature=0.2)

giga_llama_llm = LangChainLLM(llm=giga_langchain_llm)
