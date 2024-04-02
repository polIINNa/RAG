from typing import List, Tuple, Dict, Any

import numpy as np
import math
import fitz
from llama_index.legacy.schema import Document


class PyMuPDFParserWithLimit:

    def __init__(self,) -> None:
        pass

    def parse(self, fpath: str) -> List[Document]:
            pdf_reader = fitz.open(fpath)
            pages_contents, pages_bboxes, widths, heights = self.parse_pages(pdf_reader)
            doc = []

            for page_number, page_content, page_bboxes, width, height in zip(
                    range(len(pdf_reader)), pages_contents, pages_bboxes, widths, heights
                ):

                doc.append(
                    Document(
                        text=page_content,
                        metadata={
                            "file_name": fpath.split("/")[-1],
                            "page_number": page_number,
                            "bboxes": page_bboxes,
                            "width": round(width * 200 / 72.0),
                            "height": round(height * 200 / 72.0)
                        }
                    )
                )
            return doc

    def parse_pages(self, pdf_reader) -> Tuple[List[str], List[List[List[int]]], List[int], List[int]]:
        pages_contents = []
        pages_bboxes = []
        widths = []
        heights = []

        for i, page in enumerate(pdf_reader):
            lines = self.parse_page(page)

            page_content = '\n'.join([' '.join(line['text']) for line in lines if line['text'] != [' ']])
            pages_contents.append(page_content)

            page_bboxes = [line['bbox'] for line in lines if line['text'] != [' ']]
            page_bboxes = [bbox for bbox in page_bboxes if len(bbox) > 0]
            pages_bboxes.append(page_bboxes)

            widths.append(round(page.rect.width))
            heights.append(round(page.rect.height))

        return pages_contents, pages_bboxes, widths, heights

    def parse_page(self, page) -> List[Dict[str, Any]]:
        page_sorted_words = self.get_sorted_words(page)
        if len(page_sorted_words) == 0:
            lines = [{
                'text': '',
                'bbox': []
            }]

            return lines

        curr_block = page_sorted_words[0][5]
        curr_line = page_sorted_words[0][6]
        lines = []
        line_words = []
        line_bboxes = []
        height = page.rect.width
        width = page.rect.height
        for i, word in enumerate(page_sorted_words):
            x_tl, y_tl, x_br, y_br, text, block_num, line_num, word_num = word

            if curr_block != block_num:
                line_bbox = self.get_line_bboxes(line_bboxes)

                lines.append({
                    'text': line_words,
                    'bbox': line_bbox
                })

                curr_block = block_num
                curr_line = line_num
                line_words = []
                line_bboxes = []

            if curr_line != line_num:
                line_bbox = self.get_line_bboxes(line_bboxes)

                lines.append({
                    'text': line_words,
                    'bbox': line_bbox
                })

                curr_line = line_num
                line_words = []
                line_bboxes = []

            line_words.append(text)
            line_bboxes.append([x_tl, y_tl, x_br, y_br])

            if i == len(page_sorted_words) - 1:
                line_bbox = self.get_line_bboxes(line_bboxes)
                lines.append({
                            'text': line_words,
                            'bbox': line_bbox
                        })

        return lines

    @staticmethod
    def get_line_bboxes(line_bboxes: List[List[int]],) -> List[List[int]]:
        bboxes = np.array(line_bboxes)

        x_tl = round(min(bboxes[:, 0]) * 200 / 72.0)
        y_tl = round(min(bboxes[:, 1]) * 200 / 72.0)
        x_br = round(max(bboxes[:, 2]) * 200 / 72.0)
        y_br = round(max(bboxes[:, 3]) * 200 / 72.0)

        return [[x_tl, y_tl], [x_br, y_tl], [x_br, y_br], [x_tl, y_br]]

    @staticmethod
    def get_sorted_words(page) -> List[Tuple[Any]]:
        unsorted_words = list(page.get_text('words'))
        sorted_words = sorted(unsorted_words, key=lambda x: (x[-3], x[-2], x[-1]))

        return sorted_words


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

    iou = intersection_area / union_area

    return iou


def segments_lines(array):
    segments = []
    start = array[0]

    for i in range(1, len(array)):
        if array[i] != array[i-1] + 1:
            segments.append((start, array[i-1]))
            start = array[i]

    segments.append((start, array[-1]))

    return [{
        "start_line": segm[0],
        "end_line": segm[1]
    } for segm in segments]
