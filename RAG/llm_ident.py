from langchain_community.chat_models import GigaChat
# import httpx
from llama_index.legacy.llms import LangChainLLM
# from llama_index.llms.langchain import LangChainLLM

from dotenv import load_dotenv
import os

load_dotenv()

base_url = 'https://gigachat.devices.sberbank.ru/api/v1'
auth_url = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'
scope = os.environ['GIGA_SCOPE']
credentials = os.environ['GIGA_CREDENTIALS']

giga_langchain_llm = GigaChat(base_url=base_url, auth_url=auth_url, scope=scope, credentials=credentials,
                              verify_ssl_certs=False, model=os.environ['GIGA_MODEL'], profanity=False, temperature=0.2)

giga_llama_llm = LangChainLLM(llm=giga_langchain_llm)

# client = httpx.Client(proxies={'http://': os.getenv('OPENAI_PROXY'), 'https://': os.getenv('OPENAI_PROXY')})
# gpt_llm = ChatOpenAI(model='gpt-4', http_client=client)


