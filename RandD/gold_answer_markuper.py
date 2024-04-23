import json


if __name__ == '__main__':
    with open('C:/Users/ADM/OneDrive/Desktop/RAG/tagme_markup_no_html.json', 'r', encoding='utf-8') as f:
        gold_markup = json.load(f)
    answer_gold_markup = []
    for file in gold_markup:
        file_answer_gold_markup = []
        for i in range(1, 14):
            file_answer_gold_markup.append({'question': file['result'][f'q{i}'],
                                            'answer': file['result'][f'a{i}']})
        answer_gold_markup.append({'file_name': file['file_name'],
                                   'answer_gold_markup': file_answer_gold_markup})
    with open('C:/Users/ADM/OneDrive/Desktop/RAG/answer_gold_markup.json', 'w', encoding='utf-8') as f:
        json.dump(answer_gold_markup, f, ensure_ascii=False, indent=4)

