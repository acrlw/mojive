"""UI and RPC declaration edits share actual MuJoCo rebuild work."""

import mujoco
import numpy as np
import pytest

from mojive import commands as cmd
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.control.rpc import ControlService
from mojive.session import Session
from mojive.session.model_edits import ModelEditDraft

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("entry", ["ui", "rpc"])
def test_geometry_batch_compiles_once_and_preserves_physics(tmp_path, monkeypatch, entry):
    path = tmp_path / "edits.xml"
    path.write_text(
        '<mujoco><worldbody><body name="arm"><joint name="hinge"/>'
        '<geom name="box" type="box" size=".1 .2 .3"/>'
        "</body></worldbody></mujoco>"
    )
    adapter = MuJoCoAdapter(path)
    session = Session(WorkspaceAdapter(adapter), path)
    service = ControlService(session=session)
    try:
        assert session.submit(cmd.Pause()).ok
        node = next(node for node in session.nodes if node.name == "box")
        adapter._d.qpos[0], adapter._d.qvel[0] = 0.37, 0.2
        mass = float(adapter._m.body_mass[1])
        compiles, constants = [], []
        compile_model, set_constants = adapter._compile_composed_model, mujoco.mj_setConst

        def compile_counted():
            if not adapter._model_edit_batch_depth:
                compiles.append(True)
            return compile_model()

        def constants_counted(*args):
            constants.append(True)
            return set_constants(*args)

        monkeypatch.setattr(adapter, "_compile_composed_model", compile_counted)
        monkeypatch.setattr(mujoco, "mj_setConst", constants_counted)
        dimensions = ([0.3, 0.4, 0.5], [0.2, 0.3, 0.4])
        if entry == "ui":
            draft = ModelEditDraft(session)
            for size in dimensions:
                assert draft.stage(cmd.SetGeometrySize(node.node_id, size)).ok
            draft.applying = True
            result = session.apply_model_edits(draft.resolve_commands(session))
            assert result.ok, result.message
            draft.clear()
        else:
            result = service.dispatch(
                "edit_scene",
                {
                    "operations": [
                        {
                            "method": "set_geometry_size",
                            "params": {"node_id": node.node_id, "size": size},
                        }
                        for size in dimensions
                    ]
                },
            )
            assert result["ok"] and len(result["results"]) == 2
        assert compiles == [True]
        assert constants == []  # Spec compilation already calculates the final model constants.
        np.testing.assert_allclose(adapter._m.geom_size[0], dimensions[-1])
        assert adapter._m.body_mass[1] == pytest.approx(mass * 4)
        np.testing.assert_allclose(adapter._d.qpos, [0.37])
        np.testing.assert_allclose(adapter._d.qvel, [0.2])
        assert session.submit(cmd.Undo()).ok and not session.can_undo
        np.testing.assert_allclose(adapter._m.geom_size[0], [0.1, 0.2, 0.3])
        assert adapter._m.body_mass[1] == pytest.approx(mass)
        assert session.submit(cmd.Redo()).ok
        np.testing.assert_allclose(adapter._m.geom_size[0], dimensions[-1])
    finally:
        service.close()
        session.release()
