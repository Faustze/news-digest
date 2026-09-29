from urllib.parse import parse_qsl, urlencode, urlsplit

# Query parameters that only track the click and never identify the article.
TRACKING_PARAMS = frozenset(
    {"fbclid", "gclid", "yclid", "mc_cid", "mc_eid", "ref", "ref_src", "_ga"}
)


def _is_tracking_param(name: str) -> bool:
    name = name.lower()
    return name.startswith("utm_") or name in TRACKING_PARAMS


def normalize_url(url: str) -> str:
    """
    Canonical form of an article URL for identity and deduplication.

    Lowercases the scheme and host, drops the fragment, tracking parameters
    and a trailing slash, and sorts the remaining query. The path keeps its
    case: on many sites /Post and /post are different pages.
    """
    parts = urlsplit(url.strip())
    if not parts.netloc:
        return url.strip().lower()
    path = parts.path.rstrip("/")
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(k)
    )
    normalized = f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}"
    if query:
        normalized += f"?{urlencode(query)}"
    return normalized
