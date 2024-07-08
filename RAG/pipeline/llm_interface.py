from __future__ import annotations

import os
import httpx

from dotenv import load_dotenv
from langchain_community.chat_models import GigaChat


load_dotenv()

gigachat = GigaChat(base_url='https://gigachat.devices.sberbank.ru/api/v1', auth_url='https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
                    scope=os.getenv('GIGA_SCOPE'), credentials=os.getenv('GIGA_CREDENTIALS'),
                    verify_ssl_certs=False, model=os.getenv('GIGA_MODEL'),
                    profanity=False, temperature=0.0000001, http_client=httpx.Client(proxies={"http://": os.environ['OPENAI_PROXY'], "https://": os.environ['OPENAI_PROXY']}, timeout=httpx.Timeout(120.0)))
