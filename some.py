import json
import os
import re

from pdf_parser import PdfMinerParser
import summarize_splitter

if __name__ == '__main__':
    # with open('chunks_questions.json', 'r') as f:
    #     files = json.load(f)
    # for file in files:
    #     for chunk_questions in file['data']:
    #         print(chunk_questions['chunk'], '\n')
    #         print(chunk_questions['questions'], '\n\n')
    print('ПП 111.pdf'.split(".pdf")[0])
