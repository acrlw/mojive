"""Blend states for compositing into the opaque viewport target."""

PRESERVE_ALPHA = {"src_factor": "zero", "dst_factor": "one"}

ALPHA_BLEND = {
    "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
    "alpha": PRESERVE_ALPHA,
}

ADDITIVE_BLEND = {
    "color": {"src_factor": "src-alpha", "dst_factor": "one"},
    "alpha": PRESERVE_ALPHA,
}

OVERDRAW_BLEND = {
    "color": {"src_factor": "one", "dst_factor": "one"},
    "alpha": PRESERVE_ALPHA,
}
