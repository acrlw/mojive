"""Design drafts reject removed controls instead of silently discarding their edits."""

from copy import deepcopy

import pytest

from mojive.design.specification import DEFAULT, validate_spec
from mojive.design.store import SpecificationStore


@pytest.mark.parametrize(
    "part,values",
    [
        ("workspace_presets", {}),
        ("selection_caption", {"height": 28}),
        ("textarea", {"textAlign": "start"}),
        ("dialog_footer", {"rowGap": 12}),
    ],
)
def test_removed_parts_and_fields_reject_the_whole_write(tmp_path, part, values):
    store = SpecificationStore(tmp_path / "current-spec.json")
    store.write(DEFAULT, save=True)
    before = store.latest()
    invalid = deepcopy(DEFAULT)
    invalid["parts"]["button"] = {"height": 40}
    invalid["parts"][part] = values

    with pytest.raises(ValueError):
        store.write(invalid)

    assert store.latest() == before
    assert not store.live_path.exists()
    assert invalid["parts"][part] == values


def test_current_spec_preserves_all_current_component_values():
    value = deepcopy(DEFAULT)
    value["parts"]["button"] = {"height": 40, "radius": 10}
    assert validate_spec(value) == value
