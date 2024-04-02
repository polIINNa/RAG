import json
import os

from tqdm import tqdm

import pdf_parser
from pdf_parser import PyMuPDFParserWithLimit


with open("/Users/21109090/Desktop/tagme_markup_no_html.json", 'r') as f:
    tagme_markup = json.load(f)

base_path = '/Users/21109090/Desktop/госпрограмма/программы'

IOU_THR = 0.01

# ПОЛУЧЕНИЕ ГОЛД КОНТЕКСТА ПО ВСЕМ ДОКУМЕНТАМ
gold_markup = []
parser = PyMuPDFParserWithLimit()

for markup in tqdm(tagme_markup):
    markup_for_document = list(markup["result"].values())[0]
    fpath = os.path.join(base_path, markup["file_name"])
    doc = parser.parse(fpath)
    questions = []
    for index in range(13):
        pages = []
        for page_num, page in enumerate(markup_for_document["result"]["pages"]):
            context_lines = []
            for mark in page["marks"]:
                if mark["entityId"] == f'q{index+1}':
                    context_box = pdf_parser.get_points_for_rotated_rectangle(**mark["position"])
                    for i, bbox in enumerate(doc[page_num].metadata["bboxes"]):
                        iou = pdf_parser.calculate_iou(context_box, bbox)
                        if iou > IOU_THR:
                            context_lines.append(i)
            if len(context_lines) > 0:
                segments = pdf_parser.segments_lines(context_lines)
                pages.append({"page_number": page_num, "context_lines": segments})

        questions.append({
            "question": markup_for_document["result"][f"q{index+1}"],
            "context": pages
        })

    gold_markup.append({
        "file_name": markup["file_name"],
        "markup": questions
    })

with open('/Users/21109090/Desktop/госпрограмма/to_eval/gold_markup.json', 'w') as f:
    json.dump(gold_markup, f, ensure_ascii=False, indent=4)
