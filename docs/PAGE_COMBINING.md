# Page Combining Guide

Reconstruct document hierarchy across page boundaries for multi-page OCR outputs.

## Overview

When processing multi-page documents (especially tables of contents, indexes, or other structured documents), Arcane OCR generates per-page structured outputs with detected text and spatial hierarchy. The **page combining feature** merges these individual outputs into a unified document while preserving and reconstructing logical hierarchy across page breaks.

This solves key challenges:

- **Cross-page hierarchy** — text spanning or continuing across pages maintains logical nesting
- **Repeated headers** — page header lines (e.g., "contents vii", "vi contents") are filtered out
- **Section numbering** — patterns like "2 1" are recognized and formatted as "2.1"
- **Page references** — both arabic and roman numeral trailing page numbers are captured
- **Part structure** — roman numeral part prefixes are recognized (e.g., "ii geometry of two views")

## Usage

### Integration in pipeline

Add `--combine-pages` to any OCR pipeline invocation:

```bash
./scripts/run_ocr.sh \
  --input document.pdf \
  --output-dir ./output/combined \
  --combine-pages
```

With optional output base name:

```bash
./scripts/run_ocr.sh \
  --input document.pdf \
  --output-dir ./output/combined \
  --combine-pages \
  --combined-name my_combined_doc
```

### Standalone execution

Run on previously generated OCR outputs:

```bash
python3 -m src.arcane_ocr.combine_pages <input-dir> \
  --pattern "*.json" \
  --output-name combined
```

**Parameters:**

- `<input-dir>` — directory containing per-page `_structured.json` files
- `--pattern` — glob pattern for JSON files (default: `*_structured.json`)
- `--output-name` — base name for output files (default: `combined_contents`)

## Algorithm Details

### 1. Page Header Filtering

Headers repeated on each page are identified and removed. Common patterns:

- "contents xv"
- "xv contents"
- "vi acknowledgments"
- "acknowledgments vi"

Detection uses regex: `^(?:[xvi]+\s+)?contents(?:\s+[a-z]+)?$` (case-insensitive)

Removed lines are filtered during tree building, not affecting other processing.

### 2. Hierarchy Construction

Per-page structures provide `indent_level` (based on spatial x-position clustering). The page combiner uses this as the primary nesting signal across pages.

**Stack-based tree builder:**

```
For each page:
  For each line in page:
    If header: skip
    Parse section number (e.g., "2 1" → "2.1")
    Parse trailing page number (arabic or roman)
    Extract part prefixes if present
    
    Get indent level
    While stack depth > indent_level:
      pop from stack
    
    Create node with text, section, page info
    Append to current stack top's children
    Push to stack
```

This maintains context: when indent increases, the new item is a child; when indent decreases, we pop back up the tree.

### 3. Section Number Parsing

Leading numeric tokens are extracted and formatted with dots:

Input: `["2", "1", "Introduction", "15"]`
Output: section_number = "2.1", remaining = ["Introduction", "15"]

Used for display in markdown as bold prefixes ("**2.1**").

### 4. Page Number Extraction

Trailing page numbers are detected as either:

- **Roman numerals** — e.g., "vii", "xii" (matched against known roman numeral set)
- **Arabic numerals** — e.g., "15", "231" (parsed as integers at end of token list)

The algorithm walks backwards from the end of the token list:

1. Check if last token is a roman numeral → extract it
2. Else collect trailing numeric tokens

Example: `["Introduction", "to", "OCR", "15"]` → page_number = 15

### 5. Part-Level Headers

Roman numeral + following words are recognized as part headers:

- "ii geometry of two views" (roman "ii" + content)
- "iii advanced topics"

Detected as: single roman numeral token followed by non-numeric (or Roman) tokens.

### 6. Cross-Page Continuation

When a page ends at a given indent level and the next page starts with a lower indent level, the stack is wound down appropriately. Lines matching the previous indent level become siblings in the tree.

**Example:**

```
Page 1:
  Chapter 1 (indent 0)
    Section 1.1 (indent 1)
  Chapter 2 (indent 0)

Page 2:
  Section 2.1 (indent 1)    ← Becomes child of Chapter 2
  Chapter 3 (indent 0)       ← New sibling at root level
```

### 7. Output Generation

Two complementary output formats are produced:

#### JSON Output

Full tree structure with `children` arrays:

```json
{
  "children": [
    {
      "text": "Chapter 1",
      "section_number": "",
      "page_number": null,
      "page_number_roman": null,
      "source_page": "page_01",
      "children": [
        {
          "text": "Introduction",
          "section_number": "1.1",
          "page_number": 3,
          "page_number_roman": null,
          "source_page": "page_01",
          "children": []
        }
      ]
    }
  ],
  "entries_flat": [
    {
      "text": "Chapter 1",
      "section_number": "",
      "indent_level": 0,
      "page_number": null,
      "page_number_roman": null,
      "source_page": "page_01"
    },
    ...
  ]
}
```

#### Markdown Output

Nested bullet points with formatting:

```markdown
# Combined Contents

- Chapter 1
  - **1.1** Introduction ... 3
    - Background ... 5
  - **1.2** Advanced topics ... 8
- Chapter 2
  - **2.1** Framework ... 15
```

**Formatting rules:**

- Section numbers are bolded: `**2.1**`
- Page numbers are shown as trailing ` ... N`
- Roman page numbers are appended as ` ... vii`
- Indentation reflects tree depth (2 spaces per level)

## Input Format

Per-page structured JSON files must contain a `lines` array:

```json
{
  "lines": [
    {
      "text": "Chapter 1",
      "indent_level": 0
    },
    {
      "text": "1 1 Introduction 3",
      "indent_level": 1
    }
  ]
}
```

**Required fields:**

- `text` — the recognized text content
- `indent_level` — nesting depth (0 = root level)

**Optional but recommended:**

- `x_anchor` — leftmost x-position (helps verify indentation)

## Limitations

1. **OCR text quality** — The underlying OCR may produce truncated words, garbled fragments, or missed spaces. The page combiner does not correct these; it only reconstructs hierarchy from recognized text.

2. **Indent-based nesting** — Relies on spatial indentation. Documents with irregular spacing or unusual layouts may not hierarchize correctly.

3. **Section number detection** — Assumes section numbers are leading numeric tokens. Other numbering schemes (letters, mixed) may not parse as expected.

4. **Single header line per entry** — Each entry is a single text line. Multi-line entries are not merged.

5. **Page number trailing only** — Page numbers must appear at the end of the line. Embedded page numbers are not detected.

## Output Files

When combining is enabled:

- `<name>.json` — full tree structure with flat entries array
- `<name>.md` — human-readable markdown with hierarchy preserved
- Standalone mode: outputs generated in the input directory
- Pipeline mode: outputs generated in the same output directory as per-page files

## Examples

### Example 1: Table of Contents

**Input files:**  
- `page_01_structured.json`  
- `page_02_structured.json`

**page_01_structured.json:**

```json
{
  "lines": [
    {"text": "Contents", "indent_level": 0},
    {"text": "Chapter 1 Introduction 1", "indent_level": 0},
    {"text": "1 1 Background", "indent_level": 1},
    {"text": "1 2 Methods", "indent_level": 1},
    {"text": "Chapter 2 Literature", "indent_level": 0}
  ]
}
```

**page_02_structured.json:**

```json
{
  "lines": [
    {"text": "2 1 Related Work 45", "indent_level": 1},
    {"text": "2 2 State of the Art 48", "indent_level": 1},
    {"text": "Chapter 3 Results", "indent_level": 0},
    {"text": "3 1 Quantitative 61", "indent_level": 1}
  ]
}
```

**Combined JSON output:**

```json
{
  "children": [
    {
      "text": "Chapter 1 Introduction",
      "section_number": "",
      "page_number": 1,
      "source_page": "page_01",
      "children": [
        {
          "text": "Background",
          "section_number": "1.1",
          "page_number": null,
          "source_page": "page_01"
        },
        {
          "text": "Methods",
          "section_number": "1.2",
          "page_number": null,
          "source_page": "page_01"
        }
      ]
    },
    {
      "text": "Chapter 2 Literature",
      "section_number": "",
      "page_number": null,
      "source_page": "page_01",
      "children": [
        {
          "text": "Related Work",
          "section_number": "2.1",
          "page_number": 45,
          "source_page": "page_02"
        },
        {
          "text": "State of the Art",
          "section_number": "2.2",
          "page_number": 48,
          "source_page": "page_02"
        }
      ]
    },
    {
      "text": "Chapter 3 Results",
      "section_number": "",
      "page_number": null,
      "source_page": "page_02",
      "children": [
        {
          "text": "Quantitative",
          "section_number": "3.1",
          "page_number": 61,
          "source_page": "page_02"
        }
      ]
    }
  ]
}
```

**Combined Markdown output:**

```markdown
# Combined Contents

- Chapter 1 Introduction ... 1
  - **1.1** Background
  - **1.2** Methods
- Chapter 2 Literature
  - **2.1** Related Work ... 45
  - **2.2** State of the Art ... 48
- Chapter 3 Results
  - **3.1** Quantitative ... 61
```

## Integration with OCR Pipeline

The page combining module is automatically invoked during pipeline execution when:

1. `--combine-pages` flag is present
2. Two or more pages are processed

Page structures are collected during main inference loop and passed to `combine_page_structures()` after all pages complete. Output files are written to the same directory as per-page outputs.

## Performance

Page combining is a post-processing step and completes in < 100ms for typical documents. It does not require additional inference.

## Troubleshooting

### Output is mostly flat (no hierarchy)

**Symptom:** Combined output has very few nested entries.

**Causes:**

1. **Inconsistent indentation** — OCR may not consistently indent hierarchical items. Check per-page indent_level values.
2. **All text at same indent level** — Document may genuinely have flat structure.

**Solutions:**

- Inspect individual `*_structured.json` files to verify indent_level distribution
- Adjust `--pdf-scale`, `--tile-size` during initial OCR to improve spatial consistency

### Missing page numbers

**Symptom:** Combined output has `page_number: null` for entries that do have page numbers.

**Causes:**

1. **Page number not at end of line** — page combiner only detects trailing numbers
2. **OCR error** — page number was misrecognized or missing from OCR

**Solutions:**

- Check raw text in per-page JSON to see if number was recognized
- If number is embedded in middle of line, manual post-processing may be needed

### Duplicate or missing sections

**Symptom:** Some sections appear twice or are missing entirely.

**Causes:**

1. **Header filtering too aggressive** — valid section heading matched header filter regex
2. **Indent level anomaly** — OCR spatial analysis produced unexpected indent levels

**Solutions:**

- Inspect regex pattern in `combine_pages.py` (function `_is_page_header`)
- Check per-page JSON indent_level values; verify they match expected hierarchy

## Code Reference

Module: [src/arcane_ocr/combine_pages.py](../src/arcane_ocr/combine_pages.py)

Main functions:

- `combine_page_structures(page_structures, page_names)` — orchestrates combining logic
- `write_combined_outputs(combined, output_dir, base_name)` — generates JSON and Markdown
- `_build_hierarchy_from_lines(lines, page_names)` — stack-based tree construction
- `_parse_section_number(tokens)` — section number extraction
- `_extract_trailing_page_num(tokens)` — page number detection
- `_is_page_header(text)` — repeated header filtering

## See Also

- [User Guide](USERGUIDE.md) — usage examples
- [Developer Guide](DEVELOPER.md) — architecture overview
