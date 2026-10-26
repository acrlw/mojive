"""Independent persisted type choices for numeric controls and output messages."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextRole:
    family: str = "mono"
    size: int | None = None

    @classmethod
    def resolve(cls, value):
        if not isinstance(value, dict):
            return cls()
        family = value.get("family", "mono")
        size = value.get("size")
        return cls(
            family if family in ("body", "mono") else "mono",
            size if type(size) is int and 10 <= size <= 22 and size % 2 == 0 else None,
        )


@dataclass(frozen=True)
class Typography:
    numeric: TextRole = TextRole()
    output: TextRole = TextRole()

    @classmethod
    def resolve(cls, value):
        if not isinstance(value, dict):
            return cls()
        return cls(TextRole.resolve(value.get("numeric")), TextRole.resolve(value.get("output")))

    def size_for(self, parts):
        if "output_row" in parts:
            return self.output.size
        if any(part in parts for part in ("input_number", "axis_input", "input_shell")):
            return self.numeric.size
        return None


def draw_typography_settings(panel, ctx):
    """Use the same native select controls as the rest of General settings."""
    from dataclasses import replace

    from imgui_bundle import imgui

    app, ui = panel.workspace.app, panel.workspace.ui
    panel._group_heading(ctx.tr("Typography"))
    families = ("mono", "body")
    labels = [ui.font_sources[index][0].label for index in (2, 0)]
    sizes = (None, *range(10, 23, 2))
    for role, title in (("numeric", "Numeric"), ("output", "Output Messages")):
        if not panel._collapsing_header(
            ctx.tr(title) + "###settings_typography_" + role, imgui.TreeNodeFlags_.default_open
        ):
            continue
        if not panel._begin_properties("settings_typography_" + role):
            continue
        current = getattr(ui.typography, role)
        panel._property(ctx.tr("Font family"), control_part="select")
        changed, index = panel._combo("##font_" + role, families.index(current.family), labels)
        if changed:
            app.set_typography(
                replace(ui.typography, **{role: replace(current, family=families[index])})
            )
            current = getattr(ui.typography, role)
        panel._property(ctx.tr("Font size"), control_part="select")
        changed, index = panel._combo(
            "##font_size_" + role,
            sizes.index(current.size),
            [ctx.tr("Design default"), *(str(size) for size in sizes[1:])],
        )
        if changed:
            app.set_typography(
                replace(ui.typography, **{role: replace(current, size=sizes[index])})
            )
        panel._end_properties()
