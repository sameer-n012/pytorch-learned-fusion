import re
from typing import Optional

NODE_START_RE = re.compile(
    r"#\s*=+([a-zA-Z0-9_]+)\s+START=+"
)
NODE_END_RE = re.compile(
    r"#\s*=+([a-zA-Z0-9_]+)\s+END=+"
)
NODE_USER_RE = re.compile(
    r"NodeUser\(node=[A-z]+\(name='([^']+)'\)"
)

LAYOUT_RE = re.compile(r".*_layout\s*=\s*FixedLayout")
SIZES_RE = re.compile(r".*\.sizes\s*=")
LOOP_BODY_RE = re.compile(r"class\s+([a-zA-Z0-9_]+)[A-z_]*\s*:")


def parse_graph(text: str) -> dict[str, dict]:
    lines = text.splitlines()
    i = 0
    n = len(lines)

    nodes: dict[str, dict] = {}

    while i < n:
        start_match = NODE_START_RE.search(lines[i])
        if not start_match:
            i += 1
            continue

        node_name = start_match.group(1)
        i += 1

        entry_lines: list[str] = []
        while i < n:
            if NODE_END_RE.search(lines[i]):
                break
            entry_lines.append(lines[i])
            i += 1

        entry_text = "\n".join(entry_lines)

        # node users (outgoing edges in graph)
        users = sorted(set(NODE_USER_RE.findall(entry_text)))

        # node data
        node_data_lines: list[str] = []
        j = 0
        m = len(entry_lines)

        while j < m:
            line = entry_lines[j]

            # layouts
            if LAYOUT_RE.match(line):
                node_data_lines.append(line)
                j += 1
                continue

            # sizes
            if SIZES_RE.match(line):
                node_data_lines.append(line)
                j += 1
                continue

            # loop body (based on indentation)
            loop_match = LOOP_BODY_RE.match(line.strip())
            if loop_match:
                node_data_lines.append(line)
                j += 1
                while j < m and (
                    entry_lines[j].startswith("    ")
                    or entry_lines[j].strip() == ""
                ):
                    node_data_lines.append(entry_lines[j])
                    j += 1
                continue

            j += 1

        nodes[node_name] = {
            "name": node_name,
            "users": users,
            "data": "\n".join(node_data_lines).rstrip(),
        }

        # skip END line
        i += 1

    return nodes
