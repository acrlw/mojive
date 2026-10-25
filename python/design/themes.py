"""Named color presets over the shared design specification; geometry stays authored."""

import json
from copy import deepcopy

from mojive.design.resources import ROOT

PRESETS = json.loads((ROOT / "themes.json").read_text())
ROLE_TOKENS = {
    "text": "text_color",
    "secondary": "muted_color",
    "disabled": "dim_color",
    "primary": "sage_color",
    "warning": "amber_color",
    "background": "panel_color",
    "floating": "floating_color",
    "popup": "popup_color",
    "field": "input_color",
    "divider": "border_color",
}


def colors(spec):
    """Retain legacy token overrides while adding independent interaction/surface roles."""
    return {
        **PRESETS["dark"],
        **spec.get("palette", {}),
        **{role: spec["tokens"][token] for role, token in ROLE_TOKENS.items()},
    }


def theme_name(spec):
    palette = {role: value.lower() for role, value in colors(spec).items()}
    for name, values in PRESETS.items():
        if palette == {role: value.lower() for role, value in values.items()}:
            return name
    return "custom"


def themed_spec(spec, name):
    if name not in PRESETS:
        raise ValueError(f"Unknown theme: {name}")
    result = deepcopy(spec)
    result["palette"] = deepcopy(PRESETS[name])
    result["tokens"].update({token: PRESETS[name][role] for role, token in ROLE_TOKENS.items()})
    return result


def select_theme(runtime, name, *, persist=True):
    """Merge only colors, retaining concurrent layout changes and authored overrides."""
    updated = themed_spec(runtime.spec, name)
    if persist:
        envelope = runtime.store.merge(runtime.spec, updated, save=True)
        runtime.update_spec(envelope["spec"])
        runtime.poll(force=True)
    else:
        runtime.update_spec(updated)
