import json

if __name__ == '__main__':
    with open('synth_data/ПП 574.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
        print(d['relevant_chunk'])
        print(d['question'], '\n')
        print(d['data'], '\n\n')

