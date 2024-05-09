
"""
Срипт для генерации чанков из документа с программой господдержки
Чанки записываются в json файлы

"""

import os
import json

from legacy import summarize_splitter
from pipeline.pdf_parser import PdfMinerParser

if __name__ == '__main__':
    dir = '/программы/'
    files = os.listdir(dir)
    for file in files:
        print(f'ОБРАБОТКА ФАЙЛА {file}')
        docs = PdfMinerParser().parse(fpath=f'{dir}{file}')
        chunks = summarize_splitter.split(documents=docs)
        with open(f'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/chunks_prompt2/{file.split(".")[0]}.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4)
