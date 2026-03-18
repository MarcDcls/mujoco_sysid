import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import placo


class Log:
    def __init__(
        self,
        log_path: str,
        model: mujoco.MjModel,
        dt: float = 0.002,
    ):
        self.log_path = str(log_path)
        self.dt = dt

        history = placo.HistoryCollection()
        history.loadReplays(self.log_path)

        self.t0 = history.smallestTimestamp()
        self.t1 = history.biggestTimestamp()
        self.ts = np.arange(self.t0, self.t1, self.dt)

        T_world_trunk = history.pose("T_world_trunk", self.t0)
        self.base_pos = T_world_trunk[:3, 3].copy()
        self.base_pos[0] = 0.0
        self.base_pos[1] = 0.0
        self.base_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.base_quat, T_world_trunk[:3, :3].reshape(9))

        self.actuator_names: list[str] = []
        self.actuator_qpos_adr: list[int] = []
        self.tracked_actuator_ids: list[int] = []
        self.kp = np.zeros(model.nu)
        self.kd = np.zeros(model.nu)
        for actuator_id in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            self.actuator_names.append(actuator_name)

            self.kp[actuator_id] = float(history.number(f"kp:{actuator_name}", self.t0))
            self.kd[actuator_id] = float(history.number(f"kd:{actuator_name}", self.t0))

            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, actuator_name)
            self.actuator_qpos_adr.append(int(model.jnt_qposadr[joint_id]))

            if not actuator_name.startswith("Head_"):
                self.tracked_actuator_ids.append(actuator_id)

        self.q_targets = np.zeros((len(self.ts), model.nu))
        self.q_refs = np.zeros((len(self.ts), model.nu))
        self.trunk_roll_refs = np.zeros(len(self.ts))
        self.trunk_pitch_refs = np.zeros(len(self.ts))
        for i, t in enumerate(self.ts):
            T_world_trunk = history.pose("T_world_trunk", t)
            trunk_quat = np.zeros(4)
            mujoco.mju_mat2Quat(trunk_quat, T_world_trunk[:3, :3].reshape(9))
            self.trunk_roll_refs[i], self.trunk_pitch_refs[i] = self._quat_to_roll_pitch(trunk_quat)

            for actuator_id, actuator_name in enumerate(self.actuator_names):
                self.q_targets[i, actuator_id] = float(history.number(f"q_target:{actuator_name}", t))
                self.q_refs[i, actuator_id] = float(history.number(f"q:{actuator_name}", t))

    @staticmethod
    def _ground_candidate_site_names() -> list[str]:
        return [
            "left_foot",
            "left_foot_front_left",
            "left_foot_front_right",
            "left_foot_back_left",
            "left_foot_back_right",
            "right_foot",
            "right_foot_front_left",
            "right_foot_front_right",
            "right_foot_back_left",
            "right_foot_back_right",
            "left_hand",
            "right_hand",
        ]

    def _place_lowest_site_on_ground(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        mujoco.mj_forward(model, data)
        site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name) for site_name in self._ground_candidate_site_names()]
        min_z = min(float(data.site_xpos[site_id][2]) for site_id in site_ids)
        data.qpos[2] -= min_z
        mujoco.mj_forward(model, data)

    def _reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        data.qpos[self.actuator_qpos_adr] = self.q_refs[0]
        data.qpos[0:3] = self.base_pos
        data.qpos[3:7] = self.base_quat
        data.qvel[:] = 0.0
        self._place_lowest_site_on_ground(model, data)

    def _apply_constant_gains(self, model: mujoco.MjModel) -> None:
        for actuator_id in range(model.nu):
            model.actuator_gainprm[actuator_id, 0] = self.kp[actuator_id]
            model.actuator_biasprm[actuator_id, 1] = -self.kp[actuator_id]
            model.actuator_biasprm[actuator_id, 2] = -self.kd[actuator_id]

    @staticmethod
    def _quat_to_roll_pitch(quat: np.ndarray) -> tuple[float, float]:
        rot_flat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_flat, quat)
        rot = rot_flat.reshape(3, 3)

        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))

        yaw_inv = np.array(
            [
                [cos_yaw, sin_yaw, 0.0],
                [-sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        yawless_rot = yaw_inv @ rot

        r31 = float(yawless_rot[2, 0])
        r32 = float(yawless_rot[2, 1])
        r33 = float(yawless_rot[2, 2])

        pitch = float(np.arcsin(np.clip(-r31, -1.0, 1.0)))
        roll = float(np.arctan2(r32, r33))
        return roll, pitch

    def simulate(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        use_viewer: bool = False,
    ) -> tuple[float, float]:
        model.opt.timestep = self.dt
        self._apply_constant_gains(model)

        viewer_ctx = None
        joint_sse = 0.0
        trunk_angle_sse = 0.0
        nb_samples = len(self.ts)
        nb_joints = len(self.tracked_actuator_ids)

        try:
            if use_viewer:
                viewer_ctx = mujoco.viewer.launch_passive(model, data)

            self._reset(model, data)

            t_start = time.perf_counter()
            for i in range(nb_samples):
                data.ctrl[:] = self.q_targets[i]
                mujoco.mj_step(model, data)

                sse = 0.0
                for actuator_id in self.tracked_actuator_ids:
                    q_sim = data.qpos[self.actuator_qpos_adr[actuator_id]]
                    error = q_sim - self.q_refs[i, actuator_id]
                    sse += error * error
                joint_sse += sse

                roll_sim, pitch_sim = self._quat_to_roll_pitch(np.asarray(data.qpos[3:7]))
                roll_error = roll_sim - self.trunk_roll_refs[i]
                pitch_error = pitch_sim - self.trunk_pitch_refs[i]
                trunk_angle_sse += roll_error * roll_error + pitch_error * pitch_error

                if viewer_ctx is not None and viewer_ctx.is_running():
                    viewer_ctx.sync()
                    elapsed_ref = (i + 1) * self.dt
                    remaining = (t_start + elapsed_ref) - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            if viewer_ctx is not None and viewer_ctx.is_running():
                viewer_ctx.close()

        joint_mse = joint_sse / (nb_samples * nb_joints)
        trunk_angle_mse = trunk_angle_sse / (nb_samples * 2)
        return joint_mse, trunk_angle_mse


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a preprocessed log in MuJoCo.")
    parser.add_argument("--log_path", type=str, help="Path to replay log.", required=True)
    parser.add_argument("--viewer", action="store_true", help="Run with MuJoCo viewer.")
    parser.add_argument("--dt", type=float, default=0.002, help="Simulation timestep (s).")
    parser.add_argument("--model", type=str, default="../humanoid_model/k1/scene.xml", help="Path to MuJoCo scene XML.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    log = Log(
        str(Path(args.log_path)),
        model=model,
        dt=args.dt,
    )

    joint_mse, trunk_angle_mse = log.simulate(
        model=model,
        data=data,
        use_viewer=args.viewer,
    )
    joint_rmse = float(np.sqrt(joint_mse))
    trunk_angle_rmse = float(np.sqrt(trunk_angle_mse))

    print(f"joint_mse={joint_mse}")
    print(f"trunk_angle_mse={trunk_angle_mse}")
    print(f"joint_rmse={joint_rmse}")
    print(f"trunk_angle_rmse={trunk_angle_rmse}")


if __name__ == "__main__":
    main()