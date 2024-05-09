import re
from typing import List, Dict


MAX_NODE_LENGHT = 1024


def _create_point_children(point: Dict[str, str]):
    point_children = []
    subpoints = re.split('\n[a-я]{1}\)', point['point_text'])
    # Если есть подпункты - делим по подпунктам, иначе по предложениям
    if len(subpoints) > 1:
        header = subpoints[0]
        subpoints.pop(0)
        for subpoint in subpoints:
            subpoint = header + subpoint
            chunk = {'text': subpoint,
                     'page_number': point['page_number'],
                     'parent_id': point['point_id'],
                     'parent_text': point['point_text']}
            point_children.append(chunk)
    else:
        sentences = point['point_text'].split('.')
        for s in sentences:
            chunk = {'text': s,
                     'page_number': point['page_number'],
                     'parent_id': point['point_id'],
                     'parent_text': point['point_text']
                     }
            point_children.append(chunk)
    return point_children


def split(documents) -> List:
    chunks = []
    for page in documents:
        points = re.split('\n\d{1,}\.{1} ', page.text)
        for idx, point in enumerate(points):
            point_id = f'{page.metadata["file_name"]}-{page.metadata["page_number"]}-{idx}'
            point_data = {'page_number': page.metadata["page_number"],
                          'point_id': point_id,
                          'point_text': point}
            if len(point) > MAX_NODE_LENGHT:
                point_children = _create_point_children(point=point_data)
                chunks.extend(point_children)
            else:
                chunk = {
                    'text': point_data['point_text'],
                    'page_number': point_data['page_number'],
                    'parent_id': point_data['point_id'],
                    'parent_text': point_data['point_text']
                }
                chunks.append(chunk)
    return chunks

