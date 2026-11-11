"""Grounding check: is every number in the analyst's answer in the sources it was given?

The brief's hardest rule is "never invent numbers; every number must come from retrieved logs or
docs". The prompt asks for that, but a prompt is a request, not a guarantee: language models
round, add things up, or fill in a plausible figure from memory. So after generation every
number in the answer is checked against the numbers in the sources (and the question):

  * clock times ("3:20") must match exactly;
  * other numbers match if a source has the same value, allowing for rounding to the precision
    the answer used ("59 s" is supported by "59.05 s"; "59.5" is not);
  * "1,790" and "1790" are the same number; a leading minus is ignored (a value of -2.5 may be
    quoted as "2.5 lower");
  * citation markers ("[2]") and list numbering are not numbers.

Unsupported numbers trigger one repair pass (the model is told which numbers to fix); anything
still unsupported is reported to the user next to the answer, not silently shown as fact.
Also checked: every [n] refers to a source that exists.
"""

from __future__ import annotations

import re

_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_CLOCK = re.compile(r"(?<![\d:.])\d{1,3}:[0-5]\d(?![\d:])")
_NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])|(?<![\w.])-?\d+(?:\.\d+)?(?![\d])")
_LIST_MARK = re.compile(r"(?m)^\s*\d+[.)]\s")
_IDS = re.compile(r"\b\d{8}-\d{6}\w*|\b\d+[A-Z]\.\d+\b|\bs\d+\b")  # run ids, "4D.05", seeds "s42"


def cited_indices(text: str) -> list[int]:
    out = []
    for m in _CITE.finditer(text):
        out += [int(x) for x in re.split(r"\s*[,;]\s*", m.group(1))]
    return out


_GROUP_SPACE = re.compile(r"(?<=\d)[   ](?=\d{3}\b)")


def _strip(text: str) -> str:
    # "1 790" with a thin / narrow no-break space is a thousands separator (gpt-oss writes it).
    text = _GROUP_SPACE.sub("", text)
    text = _CITE.sub(" ", text)
    text = _LIST_MARK.sub(" ", text)
    return _IDS.sub(" ", text)


def numbers_in(text: str) -> tuple[set[str], list[float]]:
    """(clock times, other numbers) in ``text``."""
    text = _strip(text)
    clocks = set(_CLOCK.findall(text))
    rest = _CLOCK.sub(" ", text)
    nums = [float(m.replace(",", "")) for m in _NUM.findall(rest)]
    return clocks, nums


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def unsupported_numbers(answer: str, sources: list[str], question: str = "") -> list[str]:
    """Numbers in ``answer`` that appear in no source (nor in the question)."""
    ref_clocks, ref_nums = set(), []
    for t in [*sources, question]:
        c, n = numbers_in(t)
        ref_clocks |= c
        ref_nums += [abs(x) for x in n]
    ref_set = set(ref_nums)
    text = _strip(answer)
    bad = []
    for clock in _CLOCK.findall(text):
        if clock not in ref_clocks:
            bad.append(clock)
    for tok in _NUM.findall(_CLOCK.sub(" ", text)):
        val = abs(float(tok.replace(",", "")))
        if val in ref_set:
            continue
        d = _decimals(tok)
        # Rounded quote: a source value that rounds to what the answer wrote.
        if any(round(r, d) == val for r in ref_nums):
            continue
        bad.append(tok)
    return list(dict.fromkeys(bad))


def bad_citations(answer: str, n_sources: int) -> list[int]:
    return sorted({i for i in cited_indices(answer) if not 1 <= i <= n_sources})
