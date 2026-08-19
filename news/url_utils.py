from urllib.parse import urlsplit


def normalize_url(url: str) -> str:
    """Lowercase the host, drop the query and fragment, and drop a trailing slash."""
    parts = urlsplit(url.strip())
    if not parts.netloc:
        return url.strip().lower()
    path = parts.path.rstrip("/")
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path.lower()}"
