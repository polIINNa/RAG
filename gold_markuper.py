import json
import os
import math

from tqdm import tqdm

from pdf_parser import PdfMinerParser


with open("C:/Users/ADM/Downloads/tagme_markup_no_html.json", 'r', encoding='utf-8') as f:
    tagme_markup = json.load(f)

base_path = 'C:/Users/ADM/OneDrive/Desktop/RAG_gospodderzka/программы/'

IOU_THR = 0.01

# ПОЛУЧЕНИЕ ГОЛД КОНТЕКСТА ПО ВСЕМ ДОКУМЕНТАМ
gold_markup = []
parser = PdfMinerParser()


def get_points_for_rotated_rectangle(
        x: float, y: float, width: float, height: float, rotation: float
):
    sin_a = math.sin(rotation * math.pi / 180.0)
    cos_a = math.cos(rotation * math.pi / 180.0)
    return [
        [x, y],
        [x + width * cos_a, y + width * sin_a],
        [x - height * sin_a + width * cos_a, y + height * cos_a + width * sin_a],
        [x - height * sin_a, y + height * cos_a],

    ]


def calculate_iou(box1, box2):
    x1a, y1a, x2a, y2a = box1[0][0], box1[0][1], box1[2][0], box1[2][1]
    x1b, y1b, x2b, y2b = box2[0][0], box2[0][1], box2[2][0], box2[2][1]

    x_left = max(x1a, x1b)
    y_top = max(y1a, y1b)
    x_right = min(x2a, x2b)
    y_bottom = min(y2a, y2b)

    intersection_area = abs(max(0, x_right - x_left) * max(0, y_bottom - y_top))

    area_box1 = (x2a - x1a) * (y2a - y1a)
    area_box2 = (x2b - x1b) * (y2b - y1b)

    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / (union_area + 1e-5)

    return iou


def segments_lines(array):
    segments = []
    start = array[0]

    for i in range(1, len(array)):
        if array[i] != array[i - 1] + 1:
            segments.append((start, array[i - 1]))
            start = array[i]

    segments.append((start, array[-1]))

    return [{
        "start_line": segm[0],
        "end_line": segm[1]
    } for segm in segments]


if __name__ == '__main__':
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
                    if mark["entityId"] == f'q{index + 1}':
                        context_box = get_points_for_rotated_rectangle(**mark["position"])
                        for i, bbox in enumerate(doc[page_num].metadata["bboxes"]):
                            iou = calculate_iou(context_box, bbox)
                            if iou > IOU_THR:
                                context_lines.append(i)
                if len(context_lines) > 0:
                    segments = segments_lines(context_lines)
                    pages.append({"page_number": page_num, "context_lines": segments})

            questions.append({
                "question": markup_for_document["result"][f"q{index + 1}"],
                "context": pages
            })

        gold_markup.append({
            "file_name": markup["file_name"],
            "markup": questions
        })

    with open('C:/Users/ADM/Downloads/gold_markup.json', 'w') as f:
        json.dump(gold_markup, f, ensure_ascii=False, indent=4)
