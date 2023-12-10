from typing import List, Tuple

from scripts.entities.definitions import Condition
from scripts.entities.face import Face
from scripts.use_cases import query_matcher


def check_condition(condition: Condition, faces: List[Face], face: Face, width: int, height: int) -> bool:
    if not __is_tag_match(condition, face):
        return False

    tag_matched_faces = [f for f in faces if __is_tag_match(condition, f)]
    return __is_criteria_match(condition, tag_matched_faces, face, width, height)


def parse_tag(tag: str) -> Tuple[str, str]:
    parts = tag.split("?", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def __is_tag_match(condition: Condition, face: Face) -> bool:
    if condition.tag is None or len(condition.tag) == 0:
        return True

    condition_tag = condition.tag.lower()
    if condition_tag == "any":
        return True

    tag, query = parse_tag(condition_tag)
    face_tag = face.face_area.tag.lower() if face.face_area.tag is not None else ""
    if tag != face_tag:
        return False
    if len(query) == 0:
        return True
    return query_matcher.evaluate(query, face.face_area.attributes)


def __is_criteria_match(condition: Condition, faces: List[Face], face: Face, width: int, height: int) -> bool:
    if not condition.has_criteria():
        return True

    indices = condition.get_indices()

    if condition.is_all():
        return True

    if condition.is_left():
        return __is_left(indices, faces, face)
    if condition.is_center():
        return __is_center(indices, faces, face, width)
    if condition.is_right():
        return __is_right(indices, faces, face)
    if condition.is_top():
        return __is_top(indices, faces, face)
    if condition.is_middle():
        return __is_middle(indices, faces, face, height)
    if condition.is_bottom():
        return __is_bottom(indices, faces, face)
    if condition.is_small():
        return __is_small(indices, faces, face)
    if condition.is_large():
        return __is_large(indices, faces, face)
    return False


def __is_left(indices: List[int], faces: List[Face], face: Face) -> bool:
    sorted_faces = sorted(faces, key=lambda f: f.face_area.left)
    return sorted_faces.index(face) in indices


def __is_center(indices: List[int], faces: List[Face], face: Face, width: int) -> bool:
    sorted_faces = sorted(faces, key=lambda f: abs((f.face_area.center - width / 2)))
    return sorted_faces.index(face) in indices


def __is_right(indices: List[int], faces: List[Face], face: Face) -> bool:
    sorted_faces = sorted(faces, key=lambda f: f.face_area.right, reverse=True)
    return sorted_faces.index(face) in indices


def __is_top(indices: List[int], faces: List[Face], face: Face) -> bool:
    sorted_faces = sorted(faces, key=lambda f: f.face_area.top)
    return sorted_faces.index(face) in indices


def __is_middle(indices: List[int], faces: List[Face], face: Face, height: int) -> bool:
    sorted_faces = sorted(faces, key=lambda f: abs(f.face_area.middle - height / 2))
    return sorted_faces.index(face) in indices


def __is_bottom(indices: List[int], faces: List[Face], face: Face) -> bool:
    sorted_faces = sorted(faces, key=lambda f: f.face_area.bottom, reverse=True)
    return sorted_faces.index(face) in indices


def __is_small(indices: List[int], faces: List[Face], face: Face) -> bool:
    sorted_faces = sorted(faces, key=lambda f: f.face_area.size)
    return sorted_faces.index(face) in indices


def __is_large(indices: List[int], faces: List[Face], face: Face) -> bool:
    sorted_faces = sorted(faces, key=lambda f: f.face_area.size, reverse=True)
    return sorted_faces.index(face) in indices
