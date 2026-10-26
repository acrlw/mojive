"""Typography settings group roles without coupling their edits or collapse state."""

from types import SimpleNamespace

import pytest

from mojive.ui.localization import Language, Localizer
from mojive.ui.workspace.typography import TextRole, Typography, draw_typography_settings


@pytest.fixture
def settings():
    ui = SimpleNamespace(
        typography=Typography(TextRole("mono", 16), TextRole("body", 20)),
        font_sources=tuple((SimpleNamespace(label=name), 1) for name in ("Body", "CJK", "Mono")),
    )
    groups, tables, rows, edits = [], [], [], {}
    closed = set()

    def group(label, flags):
        caption, key = label.split("###")
        groups.append((caption, key))
        return key not in closed

    def table(key):
        tables.append(key)
        return True

    def choice(key, index, labels):
        return key in edits, edits.get(key, index)

    def save(value):
        ui.typography = value

    panel = SimpleNamespace(
        workspace=SimpleNamespace(ui=ui, app=SimpleNamespace(set_typography=save)),
        _group_heading=lambda label: None,
        _collapsing_header=group,
        _begin_properties=table,
        _end_properties=lambda: None,
        _property=lambda label, **kwargs: rows.append((tables[-1], label)),
        _combo=choice,
    )
    return SimpleNamespace(
        panel=panel, ui=ui, groups=groups, tables=tables, rows=rows, edits=edits, closed=closed
    )


@pytest.mark.parametrize("language", tuple(Language))
def test_typography_roles_have_independent_groups_and_short_field_labels(settings, language):
    localizer = Localizer(language=language)
    draw_typography_settings(settings.panel, SimpleNamespace(tr=localizer.text))
    assert settings.groups == [
        (localizer.text("Numeric"), "settings_typography_numeric"),
        (localizer.text("Output Messages"), "settings_typography_output"),
    ]
    assert settings.rows == [
        ("settings_typography_" + role, localizer.text(label))
        for role in ("numeric", "output")
        for label in ("Font family", "Font size")
    ]


@pytest.mark.parametrize("role", ("numeric", "output"))
@pytest.mark.parametrize("field", ("family", "size"))
def test_typography_edits_preserve_the_other_role_and_field(settings, role, field):
    before = settings.ui.typography
    key = "##font_" + role if field == "family" else "##font_size_" + role
    settings.edits[key] = 1 if field == "family" and role == "numeric" else 0
    draw_typography_settings(settings.panel, SimpleNamespace(tr=str))
    after = settings.ui.typography
    other_role = "output" if role == "numeric" else "numeric"
    assert getattr(after, other_role) == getattr(before, other_role)
    edited, original = getattr(after, role), getattr(before, role)
    if field == "family":
        assert edited.family != original.family
        assert edited.size == original.size
    else:
        assert edited.size is None
        assert edited.family == original.family


@pytest.mark.parametrize("role", ("numeric", "output"))
def test_collapsing_one_typography_group_preserves_the_other_role(settings, role):
    before = settings.ui.typography
    settings.closed.add("settings_typography_" + role)
    draw_typography_settings(settings.panel, SimpleNamespace(tr=str))
    other = "output" if role == "numeric" else "numeric"
    assert settings.tables == ["settings_typography_" + other]
    assert settings.ui.typography == before
