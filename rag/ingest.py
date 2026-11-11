"""PDF ingestion: traffic-engineering documents -> clean page text -> citable chunks.

Uses PyMuPDF (fast, no external tools) to read each page's text, then cleans it, because raw PDF
text is noisy in ways that hurt retrieval:
  * running headers/footers ("2009 Edition   Page 453") repeat on every page and would make
    every chunk look alike: lines that recur (digits ignored) on many pages are dropped;
  * tables of contents are dot leaders ("Overview ........ 7-1"): they match every query but
    answer none, so those lines are dropped (and a page left nearly empty is skipped);
  * scanned figures leave OCR junk (";.;::~=::.....==="): lines that are mostly not letters go;
  * a line break after a hyphen is closed up ("time-\\nof-day" -> "time-of-day"; these documents
    come from Word and are not auto-hyphenated, so a hyphen at a line end is a real one);
  * characters the PDF has no Unicode mapping for (U+FFFD, mostly dashes) become "-".

Each chunk carries the document's citation name and the PAGE AS PRINTED on it (the PDF's page
label if it has one, else the number found in the page's header/footer, e.g. "3-12"), because a
reader checks a citation against the printed page, not the PDF viewer's index. The physical page
number is kept too. The section a chunk starts in ("Section 4D.05 ...", "3.2.4 User
Characteristics"), found by its numbering pattern, is stored as metadata and prefixed to the
chunk text (the section title helps both BM25 and the embedding place a passage).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import yaml

from sim.config import PROJECT_ROOT

from .chunking import Chunk, pack_word_spans, slug

SOURCES_FILE = PROJECT_ROOT / "rag" / "sources.yaml"
INGEST_VERSION = 2  # bump when the cleaning/chunking code changes: every document is re-indexed
_DOT_LEADER = re.compile(r"\.{5,}|(?:\. ){4,}")
_PAGE_NO = re.compile(r"^(?:page\s+)?(\d{1,3}-\d{1,3}|\d{1,4}|[ivxlc]{1,6})$", re.I)
# "Section 4D.05 Application of ...", "3.2.4 User Characteristics", "7.9.4. Super-elevation Rates".
# The title may not contain digits: "10.2 Tonnes 18 Tonnes" is a table row, not a heading.
_HEADING = re.compile(r"^(?:Section\s+\d+[A-Z]\.\d+|\d+(?:\.\d+){1,3}\.?)\s+[A-Z][^.\d]{2,90}$")
_NUMBER_ONLY = re.compile(r"^\d+(?:\.\d+){1,3}\.?$")
_TITLE_LINE = re.compile(r"^[A-Z][A-Za-z ,/&()-]{2,80}$")
_HYPHEN = re.compile(r"(\w)-\n(\w)")


def load_sources() -> dict[str, dict]:
    if not SOURCES_FILE.is_file():
        return {}
    items = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8")) or []
    return {it["file"].lower(): it for it in items}


def find_pdfs(docs_dir: Path) -> list[Path]:
    return sorted(p for p in Path(docs_dir).rglob("*") if p.suffix.lower() == ".pdf" and p.is_file())


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def describe(path: Path, doc=None) -> dict:
    """Citation metadata for a PDF: the catalogue entry, else the PDF's own title, else its name."""
    src = load_sources().get(path.name.lower())
    if src:
        return {k: str(src.get(k, "")) for k in ("id", "title", "cite", "publisher", "year", "url", "license")}
    title = ""
    if doc is not None:
        title = (doc.metadata or {}).get("title") or ""
        if title.lower().startswith("microsoft word") or len(title) < 4:
            title = ""  # Word's "Microsoft Word - file.doc" is not a title
    title = title or path.stem.replace("_", " ").replace("-", " ").strip()
    return {"id": slug(path.stem)[:40], "title": title, "cite": title, "publisher": "", "year": "",
            "url": "", "license": "unknown (not in rag/sources.yaml)"}


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _is_junk(line: str) -> bool:
    s = line.strip()
    if _DOT_LEADER.search(s):
        return True
    letters = sum(c.isalpha() for c in s)
    return len(s) > 3 and letters / len(s) < 0.5


def _printed_page(lines: list[str]) -> str | None:
    """The page number printed in the header or footer, if any (last/first two lines)."""
    for line in lines[-2:][::-1] + lines[:2]:
        s = re.sub(r"\s+", " ", line.strip())
        m = _PAGE_NO.match(s) or re.search(r"\bpage\s+(\d{1,4}|\d{1,3}-\d{1,3})\s*$", s, re.I)
        if m:
            return m.group(1)
    return None


def extract_pages(path: Path) -> tuple[list[dict], dict]:
    """Pages with real content: {page, label, paragraphs: [(section, text)]}, plus document info.
    Each paragraph carries the section it belongs to (a page can contain the end of one section
    and the start of the next)."""
    import pymupdf

    doc = pymupdf.open(path)
    info = describe(path, doc)
    raw = [[ln.rstrip() for ln in doc[i].get_text("text", sort=True).splitlines()] for i in range(doc.page_count)]
    # Running headers/footers: the same line (digits ignored) near the top/bottom of many pages.
    edge = Counter()
    for lines in raw:
        nonblank = [ln for ln in lines if ln.strip()]
        for ln in set(nonblank[:3] + nonblank[-3:]):
            edge[_norm(ln)] += 1
    repeated = {k for k, n in edge.items() if n >= max(3, 0.2 * len(raw))}

    pages, section = [], ""
    for i, lines in enumerate(raw):
        nonblank = [ln for ln in lines if ln.strip()]
        label = doc[i].get_label() or _printed_page(nonblank) or str(i + 1)
        paragraphs: list[tuple[str, str]] = []
        current: list[str] = []

        def flush():
            if current:
                text = _HYPHEN.sub(r"\1-\2", "\n".join(current)).replace("\n", " ").replace("�", "-")
                paragraphs.append((section, re.sub(r"\s+", " ", text).strip()))
                current.clear()

        k = 0
        while k < len(lines):
            ln = re.sub(r"\s+", " ", lines[k].strip())
            if not ln:
                flush()
            elif _norm(ln) in repeated or _PAGE_NO.match(ln) or _is_junk(ln):
                pass
            elif _HEADING.match(ln):
                flush()
                section = ln.replace("�", "-")
                current.append(ln)
            elif _NUMBER_ONLY.match(ln) and k + 1 < len(lines) and _TITLE_LINE.match(lines[k + 1].strip()):
                flush()
                section = f"{ln} {lines[k + 1].strip()}"  # "3.2.4" on one line, the title on the next
                current.append(section)
                k += 1
            else:
                current.append(ln)
            k += 1
        flush()
        if sum(len(t.split()) for _, t in paragraphs) < 40:
            continue  # cover, blank, figure-only or table-of-contents page
        pages.append({"page": i + 1, "label": label, "paragraphs": paragraphs})
    info["pages"] = doc.page_count
    return pages, info


def pdf_chunks(path: Path, max_words: int = 200, overlap: int = 40) -> list[Chunk]:
    pages, info = extract_pages(path)
    sha = file_sha(path)
    out = []
    for p in pages:
        texts = [t for _, t in p["paragraphs"]]
        starts, n = [], 0  # (first word index, section) of each paragraph
        for sec, t in p["paragraphs"]:
            starts.append((n, sec))
            n += len(t.split())
        for j, (first, body) in enumerate(pack_word_spans(texts, max_words, overlap)):
            section = [s for w, s in starts if w <= first][-1] if starts else ""
            head = f"{info['cite']}, p. {p['label']}" + (f" ({section})" if section else "")
            out.append(Chunk(
                id=f"doc:{info['id']}:p{p['page']}:c{j}",
                text=f"{head}\n{body}",
                metadata={"source_type": "doc", "kind": "doc_page", "doc_id": info["id"], "title": info["title"],
                          "cite": info["cite"], "publisher": info["publisher"], "year": info["year"],
                          "url": info["url"], "file": path.name, "sha": sha, "page": p["page"],
                          "page_label": p["label"], "section": section,
                          "citation": f"{info['cite']}, p. {p['label']}"}))
    return out
