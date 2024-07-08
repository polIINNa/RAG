""" Скрипт от IDP для считывания pdf документов """
from typing import Tuple, Dict, Any, List

from llama_index.legacy.schema import Document
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


class PdfMinerParser:

    def __init__(self, ) -> None:
        pass

    def parse(self, fpath: str) -> List[Document]:

        pdf_reader = list(extract_pages(fpath))

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

            page_content = '\n'.join([line['text'] for line in lines if line['text'] != [' ']])
            pages_contents.append(page_content)

            page_bboxes = [line['bbox'] for line in lines if line['text'] != [' ']]
            page_bboxes = [bbox for bbox in page_bboxes if len(bbox) > 0]
            pages_bboxes.append(page_bboxes)

            widths.append(round(page.width))
            heights.append(round(page.height))

        return pages_contents, pages_bboxes, widths, heights

    def parse_page(self, page) -> List[Dict[str, Any]]:

        lines = []
        height = page.height
        width = page.width

        text_with_boxes = self.extract_text_and_boxes(page)

        for text, bbox in text_with_boxes:
            lines.append({
                "text": text,
                "bbox": self.get_line_bboxes(bbox, height)
            })

        return lines

    @staticmethod
    def get_line_bboxes(line_bbox: List[int], height: int) -> List[List[int]]:

        x_tl = round(line_bbox[0] * 200 / 72.0)
        y_tl = round((height - line_bbox[3]) * 200 / 72.0)
        x_br = round(line_bbox[2] * 200 / 72.0)
        y_br = round((height - line_bbox[1]) * 200 / 72.0)

        return [[x_tl, y_tl], [x_br, y_tl], [x_br, y_br], [x_tl, y_br]]

    @staticmethod
    def get_sorted_words(page) -> List[Tuple[Any]]:
        unsorted_words = list(page.get_text('words'))
        sorted_words = sorted(unsorted_words, key=lambda x: (x[-3], x[-2], x[-1]))

        return sorted_words

    def extract_text_and_boxes(self, page):
        text_with_boxes = []
        for element in page:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    text = text_line.get_text().strip()
                    if text:
                        text_with_boxes.append((text, text_line.bbox))
        return text_with_boxes