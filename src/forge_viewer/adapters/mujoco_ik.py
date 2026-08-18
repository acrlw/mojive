"""Damped least-squares inverse kinematics for MuJoCo adapters."""

from __future__ import annotations

import numpy as np

from .base import IkOptions, IkResult


def solve(
    mujoco,
    model,
    data,
    *,
    body: int = -1,
    site: int = -1,
    target_position,
    target_rotation,
    options: IkOptions,
) -> IkResult:
    if not options.position and not options.rotation:
        return IkResult(False, message="Enable a position or rotation target")
    if site < 0 and body <= 0:
        return IkResult(False, message="IK requires a body or site target")

    target_position = np.asarray(target_position, np.float64).reshape(3)
    target_rotation = np.asarray(target_rotation, np.float64).reshape(3, 3)
    jacp = np.zeros((3, model.nv), np.float64)
    jacr = np.zeros((3, model.nv), np.float64)
    weights = _dof_weights(model, options)
    active = weights > 0.0
    if not np.any(active):
        return IkResult(False, message="All IK joints are locked")

    position_error = rotation_error = float("inf")
    iterations = 0
    for iterations in range(1, max(1, int(options.max_iterations)) + 1):
        position, rotation = _pose(data, body, site)
        dp = target_position - position
        dr = _rotation_error(rotation, target_rotation)
        position_error = float(np.linalg.norm(dp))
        rotation_error = float(np.linalg.norm(dr))
        residual = max(
            position_error if options.position else 0.0,
            rotation_error if options.rotation else 0.0,
        )
        if residual <= max(float(options.tolerance), 1e-9):
            return IkResult(True, True, iterations - 1, position_error, rotation_error)

        jacp.fill(0.0)
        jacr.fill(0.0)
        if site >= 0:
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
        else:
            mujoco.mj_jacBody(model, data, jacp, jacr, body)
        rows = []
        errors = []
        if options.position:
            rows.append(jacp)
            errors.append(dp)
        if options.rotation:
            rows.append(jacr)
            errors.append(dr)
        jacobian = np.vstack(rows)
        error = np.concatenate(errors)
        weighted = jacobian * weights[np.newaxis, :]
        damping = max(float(options.damping), 1e-9)
        system = weighted @ weighted.T
        system.flat[:: system.shape[0] + 1] += damping * damping
        try:
            step = weights * (weighted.T @ np.linalg.solve(system, error))
        except np.linalg.LinAlgError:
            return IkResult(
                False, False, iterations, position_error, rotation_error, "IK solve failed"
            )
        step[~active] = 0.0
        limit = max(float(options.step_limit), 1e-6)
        norm = float(np.linalg.norm(step))
        if norm > limit:
            step *= limit / norm
        mujoco.mj_integratePos(model, data.qpos, step, 1.0)
        _clamp_joint_limits(model, data.qpos, mujoco)
        mujoco.mj_forward(model, data)

    return IkResult(
        True, False, iterations, position_error, rotation_error, "Iteration limit reached"
    )


def _pose(data, body: int, site: int) -> tuple[np.ndarray, np.ndarray]:
    if site >= 0:
        return (
            np.asarray(data.site_xpos[site], np.float64),
            np.asarray(data.site_xmat[site], np.float64).reshape(3, 3),
        )
    return (
        np.asarray(data.xpos[body], np.float64),
        np.asarray(data.xmat[body], np.float64).reshape(3, 3),
    )


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * sum(np.cross(current[:, axis], target[:, axis]) for axis in range(3))


def _dof_weights(model, options: IkOptions) -> np.ndarray:
    weights = np.ones(model.nv, np.float64)
    if options.joint_weights:
        for joint, weight in enumerate(options.joint_weights[: model.njnt]):
            lo = int(model.jnt_dofadr[joint])
            hi = int(model.jnt_dofadr[joint + 1]) if joint + 1 < model.njnt else model.nv
            weights[lo:hi] = max(float(weight), 0.0)
    for joint in options.locked_joints:
        if 0 <= int(joint) < model.njnt:
            lo = int(model.jnt_dofadr[int(joint)])
            hi = int(model.jnt_dofadr[int(joint) + 1]) if int(joint) + 1 < model.njnt else model.nv
            weights[lo:hi] = 0.0
    return weights


def _clamp_joint_limits(model, qpos: np.ndarray, mujoco) -> None:
    scalar = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    for joint in range(model.njnt):
        if not model.jnt_limited[joint] or int(model.jnt_type[joint]) not in scalar:
            continue
        address = int(model.jnt_qposadr[joint])
        qpos[address] = np.clip(qpos[address], *model.jnt_range[joint])
