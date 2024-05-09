import json


def add_lines(parents_datas):
    cur_page_number = 0
    cur_line = 0
    parents_datas = sorted(parents_datas, key=lambda x: x['page_place'])
    for parent_data in parents_datas:
        if parent_data['page_number'] == cur_page_number:
            parent_data['start_line'] = cur_line
            parent_data['end_line'] = cur_line + len(parent_data['text'].split('\n')) - 1
            cur_line = cur_line + len(parent_data['text'].split('\n'))
        else:
            cur_page_number = parent_data['page_number']
            cur_line = 0
            parent_data['start_line'] = 0
            parent_data['end_line'] = cur_line + len(parent_data['text'].split('\n')) - 1
            cur_line = cur_line + len(parent_data['text'].split('\n'))
    return parents_datas


if __name__ == '__main__':
    print('ПП 1598.pdf-0-0'.split('.pdf-')[1])

