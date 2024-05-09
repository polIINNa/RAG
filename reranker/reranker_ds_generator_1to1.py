"""
Скрипт для генерации датасета для обучения реранкера
Датасет генерируется на основе ранее созданных пар чанк-вопрос
"""

import json
import os
import random

from tqdm import tqdm
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd


if __name__ == '__main__':
    dir = 'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/программы'
    embed_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
    df_dict = {}
    for i in range(1536):
        df_dict[f'feature {i}'] = []
    df_dict['label'] = []
    files = os.listdir(dir)
    for file in files:
        print(f'ПРЕОБРАЗОВАНИЕ В ДАТАСЕТ ЧАНКОВ ИЗ ФАЙЛА {file}')
        with open(f'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/reranker/chunk_question_prompt2/{file.split(".")[0]}.json', 'r', encoding='utf-8') as f:
            samples = json.load(f)
        numb_samples = len(samples)
        for i in tqdm(range(numb_samples)):
            emb_question = embed_model.embed_documents(texts=[samples[i]['question']])[0]
            hard_negative_i = random.randint(0, numb_samples-1)
            while hard_negative_i == i:
                hard_negative_i = random.randint(0, numb_samples - 1)
            emb_pos_chunk = embed_model.embed_documents(texts=[samples[i]['chunk']])[0]
            for j in range(768):
                df_dict[f'feature {j}'].append(emb_question[j])
            for j in range(768, 1536):
                df_dict[f'feature {j}'].append(emb_pos_chunk[j % 768])
            df_dict['label'].append(1)
            emb_negative_chunk = embed_model.embed_documents(texts=[samples[hard_negative_i]['chunk']])[0]
            for j in range(768):
                df_dict[f'feature {j}'].append(emb_question[j])
            for j in range(768, 1536):
                df_dict[f'feature {j}'].append(emb_negative_chunk[j % 768])
            df_dict['label'].append(0)
    print('ЗАПИСЬ ДАТАФРЕЙМА В CSV')
    df = pd.DataFrame(data=df_dict)
    print(df.shape)
    train_id = int(df.shape[0]*(2/3))
    df_train = df.iloc[:train_id]
    print(df_train)
    df_test = df.iloc[train_id:]
    print(df_test)
    df_train.to_csv('C:/Users/ADM/Downloads/1to1train_prompt2.csv')
    df_test.to_csv('C:/Users/ADM/Downloads/1to1test_prompt2.csv')
