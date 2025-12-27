from __future__ import annotations

from pathlib import Path


def _parse_value(raw: str):
    val = raw.strip()
    if val == "":
        return ""
    lower = val.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def _simple_yaml_load(text: str) -> dict:
    root: dict = {}
    stack = [(0, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_val = line.strip()
        if ":" not in key_val:
            continue
        key, val = key_val.split(":", 1)
        key = key.strip()
        val = val.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            new_dict: dict = {}
            parent[key] = new_dict
            stack.append((indent + 2, new_dict))
        else:
            parent[key] = _parse_value(val)
    return root


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return _simple_yaml_load(text)
