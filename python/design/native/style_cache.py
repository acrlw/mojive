"""Bounded geometry memoization with the same field-read evidence as fresh layout."""

from functools import wraps


def prepared_geometry(method):
    """Cache immutable measurements, retaining their exact specification dependencies."""

    @wraps(method)
    def measured(style, *args, **kwargs):
        if style._compiled is None:
            return method(style, *args, **kwargs)
        key = (method.__name__, args, tuple(kwargs.items()))
        cached = style._geometry_cache.get(key)
        if cached is None:
            parent = style._collecting
            reads = style._collecting = set()
            try:
                value = method(style, *args, **kwargs)
            finally:
                style._collecting = parent
            grouped = {}
            for origin, field in reads:
                grouped.setdefault(origin, set()).add(field)
            cached = (
                value,
                tuple(reads),
                tuple((origin, frozenset(fields)) for origin, fields in grouped.items()),
            )
            if len(style._geometry_cache) >= 256:
                style._geometry_cache.clear()
            style._geometry_cache[key] = cached
        value, reads, grouped = cached
        if style._collecting is not None:
            style._collecting.update(reads)
        elif getattr(style.runtime, "track_field_reads", False):
            style.runtime.consume_groups(grouped)
        return value

    return measured
