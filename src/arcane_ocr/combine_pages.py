#!/usr/bin/env python3
"""Combine per-page structured OCR outputs into a unified document.

Designed for table-of-contents pages where hierarchy must be
reconstructed across page boundaries.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Roman numeral <-> integer (for page numbers like "vii", "xi")
_ROMAN_TO_INT = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
    "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
    "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15,
    "xvi": 16, "xvii": 17, "xviii": 18, "xix": 19, "xx": 20,
}

# Tokens that are only composed of roman numeral characters
_ROMAN_CHARS = set("ivxlcdm")


def _is_roman_numeral(token: str) -> bool:
    """Check if a token looks like a roman numeral."""
    t = token.lower().strip()
    return t in _ROMAN_TO_INT


def _roman_to_int(token: str) -> Optional[int]:
    """Convert a roman numeral token to int, or None."""
    return _ROMAN_TO_INT.get(token.lower().strip())


def _is_page_header(text: str) -> bool:
    """Return True if the line is a repeated page header like 'contents xv'."""
    stripped = text.strip().lower()
    # "contents o", "contents xv", "contents xvii", "xviii contents", "xx contents"
    if re.match(r"^(?:[xvi]+\s+)?contents(?:\s+[a-z]+)?$", stripped):
        return True
    return False


def _parse_section_number(tokens: List[str]) -> Tuple[str, int]:
    """Extract leading section number tokens for display.

    Returns (section_str, consumed_count).
    E.g. ["5", "3", "1", "planar", ...] -> ("5.3.1", 3)
         ["preface", ...] -> ("", 0)
    """
    nums: List[str] = []
    for t in tokens:
        if t.isdigit():
            nums.append(t)
        else:
            break
    section_str = ".".join(nums) if nums else ""
    return section_str, len(nums)


def _extract_trailing_page_num(tokens: List[str]) -> Tuple[Optional[int], List[str]]:
    """Extract trailing page number(s) from a token list.

    Handles arabic numerals and roman numerals at the end.
    Returns (page_number, remaining_tokens).
    """
    if not tokens:
        return None, tokens

    # First check for trailing roman numeral (e.g. "preface vii")
    last = tokens[-1].lower().strip()
    if _is_roman_numeral(last):
        return _roman_to_int(last), tokens[:-1]

    # Walk backwards collecting numeric-looking tokens
    idx = len(tokens)
    page_num = None
    while idx > 0:
        t = tokens[idx - 1]
        if t.isdigit():
            page_num = int(t)
            idx -= 1
        # Noisy digit (e.g. "1fo", "1zj") — at least 50% digits
        elif any(c.isdigit() for c in t) and sum(c.isdigit() for c in t) / len(t) >= 0.5:
            digits = "".join(c for c in t if c.isdigit())
            if digits:
                page_num = int(digits)
            idx -= 1
        else:
            break

    if page_num is not None:
        return page_num, tokens[:idx]
    return None, tokens


def parse_toc_line(text: str, indent_level: int) -> Optional[Dict]:
    """Parse a single OCR line into a structured TOC entry.

    Uses indent_level (from spatial analysis) as the primary depth signal.
    Section numbers are extracted for display but do not override depth.

    Returns dict or None if the line should be filtered.
    """
    text = text.strip()
    if not text:
        return None

    if _is_page_header(text):
        return None

    tokens = text.lower().split()
    if not tokens:
        return None

    # --- Part-level headers (roman numeral prefix at indent 0) ---
    # e.g. "ii geometry of two view 107"
    if (
        indent_level == 0
        and _is_roman_numeral(tokens[0])
        and len(tokens) > 1
        and not tokens[1].isdigit()
    ):
        part_num = _roman_to_int(tokens[0])
        rest_tokens = tokens[1:]
        page_num, title_tokens = _extract_trailing_page_num(rest_tokens)
        title = " ".join(title_tokens)
        return {
            "section_number": "",
            "part_number": part_num,
            "title": title,
            "page_number": page_num,
            "depth": 0,
            "is_part": True,
            "raw_text": text,
        }

    # --- Parse leading section numbers for display ---
    section_str, consumed = _parse_section_number(tokens)
    remaining = tokens[consumed:]

    # --- Extract trailing page number ---
    page_num, title_tokens = _extract_trailing_page_num(remaining)
    title = " ".join(title_tokens)

    # --- Depth from indent_level ---
    # indent_level 0 = top-level (chapter/part), 1 = section, 2 = subsection, etc.
    depth = indent_level

    return {
        "section_number": section_str,
        "title": title,
        "page_number": page_num,
        "depth": depth,
        "is_part": False,
        "raw_text": text,
    }


def _build_hierarchy(entries: List[Dict]) -> List[Dict]:
    """Build a nested tree from flat parsed entries.

    Uses a stack to track ancestry. Each node gets a 'children' list.
    """
    root: List[Dict] = []
    stack: List[Dict] = []

    for entry in entries:
        node = {**entry, "children": []}
        depth = entry["depth"]

        # Pop stack until we find a parent with smaller depth
        while stack and stack[-1]["depth"] >= depth:
            stack.pop()

        if stack:
            stack[-1]["children"].append(node)
        else:
            root.append(node)

        stack.append(node)

    return root


def _tree_to_md(nodes: List[Dict], level: int = 0) -> List[str]:
    """Render hierarchy tree to markdown lines."""
    lines = []
    indent = "  " * level
    for node in nodes:
        parts = []
        if node.get("is_part"):
            part_num = node.get("part_number", "")
            parts.append(f"**Part {part_num}:**")
        if node.get("section_number"):
            parts.append(f"**{node['section_number']}**")
        if node.get("title"):
            parts.append(node["title"])
        if node.get("page_number") is not None:
            parts.append(f"... {node['page_number']}")

        line_text = " ".join(parts)
        if line_text.strip():
            lines.append(f"{indent}- {line_text}")

        if node.get("children"):
            lines.extend(_tree_to_md(node["children"], level + 1))
    return lines


def _tree_to_json_entries(nodes: List[Dict]) -> List[Dict]:
    """Convert tree to JSON-serializable entries (keeping children)."""
    result = []
    for node in nodes:
        entry = {
            "section_number": node.get("section_number", ""),
            "title": node.get("title", ""),
            "page_number": node.get("page_number"),
            "depth": node.get("depth", 0),
            "raw_text": node.get("raw_text", ""),
        }
        if node.get("is_part"):
            entry["part_number"] = node.get("part_number")
            entry["is_part"] = True
        entry["children"] = _tree_to_json_entries(node.get("children", []))
        result.append(entry)
    return result


def combine_page_structures(
    page_structures: List[Dict],
    page_names: Optional[List[str]] = None,
) -> Dict:
    """Combine multiple per-page structured outputs into a single document.

    Args:
        page_structures: List of page structure dicts (each with "lines" key).
        page_names: Optional list of page names for source tracking.

    Returns:
        Combined structure dict with 'entries_flat', 'tree', and metadata.
    """
    all_parsed: List[Dict] = []
    source_pages: List[str] = []

    for page_idx, structure in enumerate(page_structures):
        page_name = page_names[page_idx] if page_names else f"page_{page_idx + 1:02d}"

        for line in structure.get("lines", []):
            text = line.get("text", "").strip()
            indent_level = int(line.get("indent_level", 0))

            parsed = parse_toc_line(text, indent_level)
            if parsed is not None:
                parsed["source_page"] = page_name
                parsed["source_line_id"] = line.get("line_id")
                all_parsed.append(parsed)
                source_pages.append(page_name)

    tree = _build_hierarchy(all_parsed)

    return {
        "entries_flat": all_parsed,
        "tree": _tree_to_json_entries(tree),
        "source_pages": sorted(set(source_pages)),
        "total_entries": len(all_parsed),
    }


def write_combined_outputs(
    combined: Dict,
    output_dir: Path,
    base_name: str = "combined_contents",
) -> Tuple[Path, Path]:
    """Write combined structure to markdown and JSON files."""
    output_dir = Path(output_dir)
    json_path = output_dir / f"{base_name}.json"
    md_path = output_dir / f"{base_name}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Combined Table of Contents",
        "",
        f"> Combined from {len(combined.get('source_pages', []))} pages: "
        + ", ".join(combined.get("source_pages", [])),
        f"> Total entries: {combined.get('total_entries', 0)}",
        "",
    ]

    tree = combined.get("tree", [])
    md_lines.extend(_tree_to_md(tree))
    md_lines.append("")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return json_path, md_path


def combine_from_directory(
    output_dir: Path,
    page_pattern: str = "*_structured.json",
    base_name: str = "combined_contents",
) -> Tuple[Path, Path]:
    """Convenience: load all structured JSONs from a directory and combine.

    Returns (json_path, md_path).
    """
    output_dir = Path(output_dir)
    json_files = sorted(output_dir.glob(page_pattern))

    if not json_files:
        raise FileNotFoundError(f"No files matching '{page_pattern}' in {output_dir}")

    page_structures = []
    page_names = []

    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            page_structures.append(json.load(f))
        page_names.append(jf.stem.replace("_structured", ""))

    combined = combine_page_structures(page_structures, page_names)
    return write_combined_outputs(combined, output_dir, base_name)


def main() -> int:
    """CLI entry point: combine structured JSONs from an output directory."""
    import argparse

    parser = argparse.ArgumentParser(description="Combine per-page OCR structured outputs")
    parser.add_argument("output_dir", help="Directory containing *_structured.json files")
    parser.add_argument("--pattern", default="*_structured.json", help="Glob pattern for JSON files")
    parser.add_argument("--name", default="combined_contents", help="Base name for combined output")
    args = parser.parse_args()

    json_path, md_path = combine_from_directory(
        Path(args.output_dir), args.pattern, args.name,
    )
    print(f"[INFO] Combined JSON: {json_path}")
    print(f"[INFO] Combined Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
