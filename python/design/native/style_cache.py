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
            cached = (value, *_dependency_evidence(style, reads))
            if len(style._geometry_cache) >= 256:
                # New scene rows must not discard the still-visible rows and
                # field measurements. Evict one cold entry, not the whole cache.
                style._geometry_cache.popitem(last=False)
            style._geometry_cache[key] = cached
        else:
            style._geometry_cache.move_to_end(key)
        value, reads, grouped = cached
        if style._collecting is not None:
            style._collecting.update(reads)
        elif getattr(style.runtime, "track_field_reads", False):
            style.runtime.consume_groups(grouped)
        return value

    return measured


def _dependency_evidence(style, reads):
    # Different row positions usually read the same specification fields. Share
    # this immutable evidence within one frozen style, including exact branch reads.
    key = frozenset(reads)
    cache = style._geometry_dependencies
    evidence = cache.get(key)
    if evidence is None:
        grouped = {}
        for origin, field in reads:
            grouped.setdefault(origin, set()).add(field)
        evidence = (
            tuple(reads),
            tuple((origin, frozenset(fields)) for origin, fields in grouped.items()),
        )
        if len(cache) >= 64:
            cache.popitem(last=False)
        cache[key] = evidence
    else:
        cache.move_to_end(key)
    return evidence
