import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import placo


def set_joints_from_history(model: mujoco.MjModel, data: mujoco.MjData, history: placo.HistoryCollection, t: float) -> None:
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        q = float(history.number(f"q:{joint_name}", t))
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = q


def set_base_from_history(data: mujoco.MjData, history: placo.HistoryCollection, t: float) -> None:
    # Set base position
    T_world_trunk = history.pose("T_world_trunk", t)
    data.qpos[0:3] = T_world_trunk[:3, 3]

    # Set base orientation
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T_world_trunk[:3, :3].reshape(9))
    data.qpos[3:7] = quat

    # Center base on world XY plane
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0


def support_foot_sites(support_foot: str) -> list[str]:
    if support_foot == "right":
        return [
            "right_foot",
            "right_foot_front_left",
            "right_foot_front_right",
            "right_foot_back_left",
            "right_foot_back_right",
        ]
    return [
        "left_foot",
        "left_foot_front_left",
        "left_foot_front_right",
        "left_foot_back_left",
        "left_foot_back_right",
    ]


def place_support_foot_on_ground(model: mujoco.MjModel, data: mujoco.MjData, support_foot: str = "left") -> None:
    mujoco.mj_forward(model, data)

    min_foot_z = float("inf")
    for site_name in support_foot_sites(support_foot):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            continue
        min_foot_z = min(min_foot_z, float(data.site_xpos[site_id][2]))

    if np.isfinite(min_foot_z):
        data.qpos[2] -= min_foot_z
        mujoco.mj_forward(model, data)
        return

    raise RuntimeError(f"No support-foot sites found for '{support_foot}'")


def apply_joint_commands(model: mujoco.MjModel, data: mujoco.MjData, history: placo.HistoryCollection, t: float) -> None:
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if actuator_name is None:
            continue

        data.ctrl[actuator_id] = float(history.number(f"q_target:{actuator_name}", t))


def apply_constant_gains_from_history(model: mujoco.MjModel, history: placo.HistoryCollection, t: float) -> None:
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if actuator_name is None:
            continue

        kp = float(history.number(f"kp:{actuator_name}", t))
        kd = float(history.number(f"kd:{actuator_name}", t))

        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd


def tracking_error_sse(model: mujoco.MjModel, data: mujoco.MjData, history: placo.HistoryCollection, t: float) -> float:
    error = 0.0
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None or joint_name.startswith("Head_"):
            continue

        q_ref = float(history.number(f"q:{joint_name}", t))
        q_sim = float(data.qpos[model.jnt_qposadr[joint_id]])
        diff = q_sim - q_ref
        error += diff * diff

    return error


def tracked_joint_count(model: mujoco.MjModel) -> int:
    count = 0
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None or joint_name.startswith("Head_"):
            continue
        count += 1
    return count


def simulate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    history: placo.HistoryCollection,
    dt: float = 0.005,
    use_viewer: bool = False,
    support_foot: str = "left",
) -> tuple[float, float]:
    model.opt.timestep = dt

    t0 = history.smallestTimestamp()
    t1 = history.biggestTimestamp()
    ts = np.arange(t0, t1, dt)

    apply_constant_gains_from_history(model, history, t0)
    set_joints_from_history(model, data, history, t0)
    set_base_from_history(data, history, t0)
    place_support_foot_on_ground(model, data, support_foot=support_foot)
    data.qvel[:] = 0.0

    viewer_ctx = None
    score_sse = 0.0
    nb_samples = 0
    try:
        if use_viewer:
            viewer_ctx = mujoco.viewer.launch_passive(model, data)

        t_wall_start = time.perf_counter()
        for i, t in enumerate(ts):
            apply_joint_commands(model, data, history, t)
            mujoco.mj_step(model, data)
            score_sse += tracking_error_sse(model, data, history, t)
            nb_samples += 1

            if viewer_ctx is not None and viewer_ctx.is_running():
                viewer_ctx.sync()

                elapsed_ref = (i + 1) * dt
                remaining = (t_wall_start + elapsed_ref) - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if viewer_ctx is not None and viewer_ctx.is_running():
            viewer_ctx.close()

    nb_joints = tracked_joint_count(model)
    if nb_samples == 0 or nb_joints == 0:
        return 0.0, 0.0

    score_mse = score_sse / (nb_samples * nb_joints)
    score_rmse = float(np.sqrt(score_mse))
    return score_mse, score_rmse


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a HistoryCollection log in MuJoCo.")
    parser.add_argument("--log_path", type=str, help="Path to replay log.", required=True)
    parser.add_argument("--viewer", action="store_true", help="Run with MuJoCo viewer.")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep (s).")
    parser.add_argument("--support-foot", choices=["left", "right"], default="left", help="Foot to place on ground at start.")
    parser.add_argument("--model", type=str, default="../humanoid_model/k1/scene.xml", help="Path to MuJoCo scene XML.")
    args = parser.parse_args()

    history = placo.HistoryCollection()
    history.loadReplays(args.log_path)

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    score_mse, score_rmse = simulate(
        model=model,
        data=data,
        history=history,
        dt=args.dt,
        use_viewer=args.viewer,
        support_foot=args.support_foot,
    )
    print(f"score_mse={score_mse}")
    print(f"score_rmse={score_rmse}")


if __name__ == "__main__":
    main()