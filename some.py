import json

if __name__ == '__main__':
    with open('chunks_questions/ПП 26.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    test = data[0]
    print(test['questions'])


