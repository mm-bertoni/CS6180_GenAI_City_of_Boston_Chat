"""Tests for ui/formatting.py. Run from the repo root:

    venv/Scripts/python.exe -m pytest ui/tests -q

No Streamlit import anywhere in the chain, which is the point - these rules are
testable without starting a server.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.formatting import (
    citation_label,
    format_event_datetime,
    format_score,
    group_sources,
    linkify_citations,
    snippet,
    source_kind,
    source_label,
    strip_chunk_header,
)

PAGE_TEXT_SOURCE = {
    "notice_id": "16492911",
    "title": "Boston Retirement Board Meeting",
    "url": "https://www.boston.gov/public-notices/16492911",
    "source_type": "page_text",
    "file_label": None,
    "event_datetime": "2026-07-22T13:00:00Z",
    "score": 0.4602,
    "text": "Boston Retirement Board Meeting | event 2026-07-22T13:00:00Z\nAdministrative Session",
}

PDF_SOURCE = {
    "notice_id": "16492911",
    "title": "Boston Retirement Board Meeting",
    "url": "https://www.boston.gov/public-notices/16492911",
    "source_type": "pdf",
    "file_label": "Revised Official Filed Posting",
    "event_datetime": "2026-07-22T13:00:00Z",
    "score": 0.4676,
    "text": "Boston Retirement Board Meeting | event 2026-07-22T13:00:00Z\nBOSTON RETIREMENT BOARD\nTO:\nALEX GEOURNTAS, CITY CLERK",
}


#timezone conversion

def test_utc_converts_to_boston_local_across_the_dst_boundary():
    assert format_event_datetime("2026-07-22T13:00:00Z") == "Jul 22, 2026 at 09:00 AM EDT"
    assert format_event_datetime("2026-11-18T14:00:00Z") == "Nov 18, 2026 at 09:00 AM EST"
    assert format_event_datetime("2026-12-16T14:00:00Z") == "Dec 16, 2026 at 09:00 AM EST"


def test_missing_datetime_is_none_and_garbage_passes_through():
    assert format_event_datetime(None) is None
    assert format_event_datetime("") is None
    assert format_event_datetime("not a date") == "not a date"


# chunk header

def test_injected_header_line_is_stripped():
    assert strip_chunk_header(PAGE_TEXT_SOURCE["text"]) == "Administrative Session"
    assert strip_chunk_header(PDF_SOURCE["text"]).startswith("BOSTON RETIREMENT BOARD")


def test_text_without_a_header_keeps_its_first_line():
    body = "Notice of a public hearing\nheld at City Hall"
    assert strip_chunk_header(body) == body


def test_snippet_truncates_on_a_word_boundary():
    out = snippet("word " * 200, limit=40)
    assert len(out) <= 43 and out.endswith("...") and "wor..." not in out


# file_label guard 
def test_page_text_sources_get_a_readable_label():
    assert source_label(PAGE_TEXT_SOURCE) == "notice webpage"
    assert source_label(PDF_SOURCE) == "Revised Official Filed Posting"
    assert source_kind(PAGE_TEXT_SOURCE) == "Notice webpage"
    assert source_kind(PDF_SOURCE) == "PDF"


# citation numbering 

def test_same_notice_groups_without_losing_either_entry():
    groups = group_sources([PAGE_TEXT_SOURCE, PDF_SOURCE])
    assert len(groups) == 1
    notice_id, items = groups[0]
    assert notice_id == "16492911"
    assert [i for i, _ in items] == [1, 2]


def test_every_original_index_survives_grouping():
    other = dict(PAGE_TEXT_SOURCE, notice_id="16492931", title="Zoning Hearing")
    sources = [PAGE_TEXT_SOURCE, other, PDF_SOURCE, dict(other, source_type="pdf")]
    groups = group_sources(sources)
    emitted = sorted(i for _, items in groups for i, _ in items)
    assert emitted == [1, 2, 3, 4]
    assert dict(groups)["16492911"] == [(1, sources[0]), (3, sources[2])]


def test_citation_label_pluralises():
    assert citation_label([1]) == "source 1"
    assert citation_label([1, 2]) == "sources 1, 2"


#inline citations

SOURCES = [PAGE_TEXT_SOURCE, PDF_SOURCE]
URL = "https://www.boston.gov/public-notices/16492911"


def test_both_bracket_styles_become_links_keeping_their_delimiters():
    # The model emits [1] and (1) interchangeably for identical prompts.
    assert linkify_citations("Scheduled for July 22 (1).", SOURCES) == (
        f'Scheduled for July 22 ([1]({URL} "Boston Retirement Board Meeting - '
        f'Notice webpage: notice webpage - Jul 22, 2026 at 09:00 AM EDT - '
        f'Click to open on boston.gov")).'
    )
    assert linkify_citations("Scheduled [2].", SOURCES).startswith("Scheduled [[2](")


def test_grouped_markers_link_each_number_separately():
    out = linkify_citations("Both agree (1, 2).", SOURCES)
    assert out.count("](") == 2
    assert out.startswith("Both agree (") and out.endswith(").")


def test_tooltip_names_the_notice_and_the_local_time():
    out = linkify_citations("Text (1).", SOURCES)
    assert "Boston Retirement Board Meeting" in out
    assert "Jul 22, 2026 at 09:00 AM EDT" in out   # local, not the raw 13:00Z
    assert "Notice webpage: notice webpage" in out


def test_out_of_range_and_incidental_parens_are_left_alone():
    assert linkify_citations("Claim (7).", SOURCES) == "Claim (7)."
    assert linkify_citations("It ran (2 hours).", SOURCES) == "It ran (2 hours)."


def test_no_sources_means_no_rewriting():
    assert linkify_citations("Nothing found (1).", []) == "Nothing found (1)."


def test_quotes_in_a_title_cannot_break_out_of_the_tooltip():
    quoted = dict(PAGE_TEXT_SOURCE, title='The "Big Dig" Hearing')
    out = linkify_citations("Text (1).", [quoted])
    assert out.count('"') == 2


# score

def test_score_is_labelled_as_a_distance_not_a_percentage():
    out = format_score(0.4602)
    assert "0.4602" in out and "distance" in out and "%" not in out