
import re
from datetime import datetime
from zoneinfo import ZoneInfo

BOSTON_TZ = "America/New_York"

CHUNK_HEADER = re.compile(r"^.+\|\s*event\s+\S+\s*$")

SOURCE_KINDS = {"pdf": "PDF", "page_text": "Notice webpage"}

CITATION = re.compile(r"[\[(]\s*\d+(?:\s*(?:,|;|and)\s*\d+)*\s*[\])]")


def format_event_datetime(raw, tz=BOSTON_TZ):
    if not raw:
        return None
    try:
        utc = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return utc.astimezone(ZoneInfo(tz)).strftime("%b %d, %Y at %I:%M %p %Z")
    except (ValueError, TypeError):
        return str(raw)


def strip_chunk_header(text):
    if not text:
        return ""
    head, _, rest = text.partition("\n")
    if CHUNK_HEADER.match(head.strip()):
        return rest.strip()
    return text.strip()


def snippet(text, limit=320):
    body = " ".join(strip_chunk_header(text).split())
    if len(body) <= limit:
        return body
    return body[:limit].rsplit(" ", 1)[0] + "..."


def source_label(src):
    return src.get("file_label") or "notice webpage"


def source_kind(src):
    kind = src.get("source_type")
    return SOURCE_KINDS.get(kind, kind or "unknown")


def citation_label(indices):
    numbers = ", ".join(str(i) for i in indices)
    return f"source {numbers}" if len(indices) == 1 else f"sources {numbers}"


def format_score(score):
    return f"{score:.4f} (distance, lower is better)"


def citation_tooltip(src):
    parts = [src.get("title") or "Untitled notice", f"{source_kind(src)}: {source_label(src)}"]
    when = format_event_datetime(src.get("event_datetime"))
    if when:
        parts.append(when)
    parts.append("Click to open on boston.gov")
    # Markdown link titles are double-quoted, so a quote in a notice title would
    # end the title early and leak markup into the page.
    return " - ".join(parts).replace('"', "'")


def linkify_citations(text, sources):
    if not text or not sources:
        return text

    def replace(match):
        marker = match.group(0)
        numbers = [int(n) for n in re.findall(r"\d+", marker)]
        links = []
        for number in numbers:
            if not 1 <= number <= len(sources):
                return marker            # out of range: not ours to touch
            src = sources[number - 1]
            url = src.get("url")
            if not url:
                return marker            # nothing to link to
            links.append(f'[{number}]({url} "{citation_tooltip(src)}")')
        return f"{marker[0]}{', '.join(links)}{marker[-1]}"

    return CITATION.sub(replace, text)


def group_sources(sources):
    groups = {}
    for index, src in enumerate(sources, 1):
        # Fall back to the position so a source missing notice_id still gets shown.
        key = src.get("notice_id") or f"_{index}"
        groups.setdefault(key, []).append((index, src))
    return list(groups.items())