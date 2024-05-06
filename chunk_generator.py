import json
import os

from pdf_parser import PdfMinerParser
import summarize_splitter


if __name__ == '__main__':
    output = []
    dir = '/Users/21109090/Desktop/RAG_gospodderzka/программы/'
    files = os.listdir(dir)
    for file in files:
        print(f'ОБРАБОТКА ФАЙЛА {file}')
        docs = PdfMinerParser().parse(fpath=f'{dir}{file}')
        chunks = summarize_splitter.split(documents=docs)
        print('СОХРАНЕНИЕ ЧАНКОВ В ФАЙЛ')
        with open(f'/Users/21109090/Desktop/RAG_gospodderzka/chunks/{file.split(".pdf")[0]}.json') as f:
            json.dump(chunks, f, indent=4)

