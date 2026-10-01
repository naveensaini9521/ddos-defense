"""Never-block list (gateway, DNS, admin)."""

WHITELIST: set[str] = set()


def is_whitelisted(ip: str) -> bool:
    return ip in WHITELIST
