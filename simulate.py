import argparse
import time
from pathlib import Path
import re

import mujoco
import mujoco.viewer
import numpy as np
import placo
import onnxruntime as ort

class Log:
    POLICY_MASKED_DOFS = {"Head_Yaw", "Head_Pitch"}

    def __init__(
        self,
        log_path: str,
        model: mujoco.MjModel,
        tracked_joints: list[str],
        dt: float = 0.002,
        agent_path: str | None = None,
        infer_frequency: float = 0.0,
        joint_obs_delay_ms: float = 0.0,
        imu_obs_delay_ms: float = 0.0,
    ):
        self.log_path = str(log_path)
        self.dt = dt

        self.agent = None
        if agent_path is not None:
            self.agent = ort.InferenceSession(agent_path)
        self.infer_frequency = infer_frequency
        self.last_infer_time = 0.0
        self.joint_obs_delay_ms = max(0.0, joint_obs_delay_ms)
        self.imu_obs_delay_ms = max(0.0, imu_obs_delay_ms)
        self.joint_obs_delay_steps = int(round(self.joint_obs_delay_ms * 1e-3 / self.dt))
        self.imu_obs_delay_steps = int(round(self.imu_obs_delay_ms * 1e-3 / self.dt))
        self.policy_input_name: str | None = None
        self.policy_output_name: str | None = None
        self.policy_observation_names: list[str] = []
        self.policy_joint_names: list[str] = []
        self.policy_joint_default_position: list[float] = []
        self.policy_action_scale: float = 1.0
        self.policy_masked_joint_names: list[str] = []
        self.policy_masked_actuator_ids: list[int] = []
        self.policy_last_action: np.ndarray | None = None
        self.joint_pos_obs_buffer: list[np.ndarray] = []
        self.joint_vel_obs_buffer: list[np.ndarray] = []
        self.imu_gyro_obs_buffer: list[np.ndarray] = []
        self.imu_gravity_obs_buffer: list[np.ndarray] = []

        self.imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/imu")

        if self.agent is not None:
            self._load_policy_metadata()

        history = placo.HistoryCollection()
        history.loadReplays(self.log_path)
        self.history = history

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
        self.actuator_qvel_adr: list[int] = []
        self.tracked_actuator_ids: list[int] = []
        self.kp = np.zeros(model.nu)
        self.kd = np.zeros(model.nu)
        for actuator_id in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            self.actuator_names.append(actuator_name)

            gain_time = 0.5 * (self.t0 + self.t1)
            self.kp[actuator_id] = float(history.number(f"kp:{actuator_name}", gain_time))
            self.kd[actuator_id] = float(history.number(f"kd:{actuator_name}", gain_time))

            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, actuator_name)
            self.actuator_qpos_adr.append(int(model.jnt_qposadr[joint_id]))
            self.actuator_qvel_adr.append(int(model.jnt_dofadr[joint_id]))

            for tracked_joint in tracked_joints:
                if tracked_joint in actuator_name:
                    self.tracked_actuator_ids.append(actuator_id)
                    break

        self.q_targets = np.zeros((len(self.ts), model.nu))
        self.q_refs = np.zeros((len(self.ts), model.nu))
        self.trunk_roll_refs = np.zeros(len(self.ts))
        self.trunk_pitch_refs = np.zeros(len(self.ts))
        self.cmd_vx = np.zeros(len(self.ts))
        self.cmd_vy = np.zeros(len(self.ts))
        self.cmd_vyaw = np.zeros(len(self.ts))
        self.cmd_cos_phase = np.zeros(len(self.ts))
        self.cmd_sin_phase = np.zeros(len(self.ts))
        for i, t in enumerate(self.ts):
            T_world_trunk = history.pose("T_world_trunk", t)
            trunk_quat = np.zeros(4)
            mujoco.mju_mat2Quat(trunk_quat, T_world_trunk[:3, :3].reshape(9))
            self.trunk_roll_refs[i], self.trunk_pitch_refs[i] = self._quat_to_roll_pitch(trunk_quat)

            self.cmd_vx[i] = float(history.number("policy_vx", t))
            self.cmd_vy[i] = float(history.number("policy_vy", t))
            self.cmd_vyaw[i] = float(history.number("policy_vyaw", t))
            self.cmd_cos_phase[i] = float(history.number("policy_cos_phase", t))
            self.cmd_sin_phase[i] = float(history.number("policy_sin_phase", t))

            for actuator_id, actuator_name in enumerate(self.actuator_names):
                self.q_targets[i, actuator_id] = float(history.number(f"q_target:{actuator_name}", t))
                self.q_refs[i, actuator_id] = float(history.number(f"q:{actuator_name}", t))

        if self.agent is not None:
            self._build_policy_joint_mapping(model)

    @staticmethod
    def _split_metadata_values(raw_value: str) -> list[str]:
        values = []
        for token in re.split(r",", raw_value):
            value = token.strip().strip("()\n\t\r'\"")
            if value:
                values.append(value)
        return values

    def _load_policy_metadata(self) -> None:
        if self.agent is None:
            return

        model_meta = self.agent.get_modelmeta().custom_metadata_map
        required = [
            "joint_names",
            "default_joint_pos",
            "observation_names",
            "action_scale",
        ]
        for key in required:
            if key not in model_meta:
                raise RuntimeError(f"Policy metadata missing required field '{key}'")

        self.policy_joint_names = self._split_metadata_values(model_meta["joint_names"])
        self.policy_joint_default_position = [float(v) for v in self._split_metadata_values(model_meta["default_joint_pos"])]
        self.policy_observation_names = self._split_metadata_values(model_meta["observation_names"])
        self.policy_action_scale = float(model_meta["action_scale"])

        if len(self.policy_joint_names) != len(self.policy_joint_default_position):
            raise RuntimeError("Policy metadata mismatch: joint_names and default_joint_pos sizes differ")

        self.policy_input_name = self.agent.get_inputs()[0].name
        self.policy_output_name = self.agent.get_outputs()[0].name

    def _build_policy_joint_mapping(self, model: mujoco.MjModel) -> None:
        if self.agent is None:
            return

        output_size = int(self.agent.get_outputs()[0].shape[-1])
        actuator_name_to_id: dict[str, int] = {}
        for actuator_id in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            actuator_name_to_id[actuator_name] = actuator_id

        candidate_joint_names = [joint_name for joint_name in self.policy_joint_names if joint_name in actuator_name_to_id]

        excluded_joints: set[str] = set(self.POLICY_MASKED_DOFS)

        if excluded_joints:
            self.policy_masked_joint_names = [
                joint_name for joint_name in candidate_joint_names if joint_name not in excluded_joints
            ]
        elif output_size == len(candidate_joint_names):
            self.policy_masked_joint_names = list(candidate_joint_names)
        else:
            raise RuntimeError(
                "Unable to infer policy controlled joints: metadata/config does not provide masked dofs and output size doesn't match candidate joints"
            )

        self.policy_masked_actuator_ids = []
        for joint_name in self.policy_masked_joint_names:
            if joint_name not in actuator_name_to_id:
                raise RuntimeError(f"Policy joint '{joint_name}' is not an actuator in the MuJoCo model")
            self.policy_masked_actuator_ids.append(actuator_name_to_id[joint_name])

        self.policy_last_action = np.zeros(output_size, dtype=np.float32)

        if len(self.policy_masked_joint_names) != output_size:
            raise RuntimeError(
                f"Unable to map policy output ({output_size}) to controlled joints ({len(self.policy_masked_joint_names)})"
            )

    @staticmethod
    def _quat_to_rotation(quat: np.ndarray) -> np.ndarray:
        rot_flat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_flat, quat)
        return rot_flat.reshape(3, 3)

    @staticmethod
    def _read_delayed(buffer: list[np.ndarray], delay_steps: int) -> np.ndarray:
        if not buffer:
            raise RuntimeError("Observation delay buffer is empty")
        delayed_index = max(0, len(buffer) - 1 - delay_steps)
        return buffer[delayed_index]

    def _update_observation_buffers(self, data: mujoco.MjData) -> None:
        if not self.policy_masked_actuator_ids:
            return

        joint_pos = np.asarray(
            [data.qpos[self.actuator_qpos_adr[actuator_id]] for actuator_id in self.policy_masked_actuator_ids],
            dtype=np.float32,
        )
        joint_vel = np.asarray(
            [data.qvel[self.actuator_qvel_adr[actuator_id]] for actuator_id in self.policy_masked_actuator_ids],
            dtype=np.float32,
        )

        rot_world_trunk = np.asarray(data.site_xmat[self.imu_site_id]).reshape(3, 3).copy()
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_trunk = rot_world_trunk.T @ gravity_world
        gyro = np.asarray(data.sensor("imu_ang_vel").data.copy(), dtype=np.float32)

        self.joint_pos_obs_buffer.append(joint_pos)
        self.joint_vel_obs_buffer.append(joint_vel)
        self.imu_gyro_obs_buffer.append(gyro)
        self.imu_gravity_obs_buffer.append(gravity_trunk)

    def _build_policy_observation(self, data: mujoco.MjData, sample_idx: int) -> np.ndarray:
        if self.policy_last_action is None:
            raise RuntimeError("Policy action state is not initialized")

        delayed_joint_pos = self._read_delayed(self.joint_pos_obs_buffer, self.joint_obs_delay_steps)
        delayed_joint_vel = self._read_delayed(self.joint_vel_obs_buffer, self.joint_obs_delay_steps)
        delayed_gyro = self._read_delayed(self.imu_gyro_obs_buffer, self.imu_obs_delay_steps)
        delayed_gravity = self._read_delayed(self.imu_gravity_obs_buffer, self.imu_obs_delay_steps)

        values: list[float] = []
        for observation_name in self.policy_observation_names:
            if observation_name == "base_lin_vel":
                values.extend([0.0, 0.0, 0.0])
            elif observation_name == "base_ang_vel":
                values.extend([
                    float(delayed_gyro[0]),
                    float(delayed_gyro[1]),
                    float(delayed_gyro[2]),
                ])
            elif observation_name == "projected_gravity":
                values.extend([
                    float(delayed_gravity[0]),
                    float(delayed_gravity[1]),
                    float(delayed_gravity[2]),
                ])
            elif observation_name == "joint_pos":
                default_position_by_joint = {
                    name: self.policy_joint_default_position[idx] for idx, name in enumerate(self.policy_joint_names)
                }
                for idx, _ in enumerate(self.policy_masked_actuator_ids):
                    joint_name = self.policy_masked_joint_names[idx]
                    q_default = float(default_position_by_joint[joint_name])
                    values.append(float(delayed_joint_pos[idx]) - q_default)
            elif observation_name == "joint_vel":
                for idx, _ in enumerate(self.policy_masked_actuator_ids):
                    values.append(float(delayed_joint_vel[idx]))
            elif observation_name == "actions":
                values.extend([float(x) for x in self.policy_last_action])
            elif observation_name == "command":
                values.extend([
                    float(self.cmd_vx[sample_idx]),
                    float(self.cmd_vy[sample_idx]),
                    float(self.cmd_vyaw[sample_idx]),
                ])
            elif observation_name == "reference_phase":
                values.extend([
                    float(self.cmd_cos_phase[sample_idx]),
                    float(self.cmd_sin_phase[sample_idx]),
                ])
            else:
                raise RuntimeError(f"Unsupported observation term for mujoco_sysid simulation: {observation_name}")

        return np.asarray(values, dtype=np.float32)

    def _compose_policy_targets(self, sample_idx: int, action: np.ndarray) -> np.ndarray:
        ctrl = np.array(self.q_targets[sample_idx], copy=True)
        default_position_by_joint = {
            name: self.policy_joint_default_position[idx] for idx, name in enumerate(self.policy_joint_names)
        }

        for idx, actuator_id in enumerate(self.policy_masked_actuator_ids):
            joint_name = self.policy_masked_joint_names[idx]
            joint_default = default_position_by_joint[joint_name]
            ctrl[actuator_id] = joint_default + self.policy_action_scale * float(action[idx])

        return ctrl

    def _infer_policy_action(self, data: mujoco.MjData, sample_idx: int) -> np.ndarray:
        if self.agent is None or self.policy_input_name is None or self.policy_output_name is None:
            raise RuntimeError("Policy inference requested but ONNX session is not initialized")

        observation = self._build_policy_observation(data, sample_idx)
        action = self.agent.run(
            [self.policy_output_name],
            {self.policy_input_name: observation.reshape(1, -1)},
        )[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return action

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
        data.ctrl[:] = self.q_targets[0]
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
            self.last_infer_time = t_start
            infer_period = (1.0 / self.infer_frequency) if self.infer_frequency > 0.0 else 0.0
            next_infer_elapsed = 0.0
            current_ctrl = np.array(self.q_targets[0], copy=True)
            for i in range(nb_samples):
                if self.agent is not None:
                    self._update_observation_buffers(data)

                elapsed = i * self.dt
                should_infer = self.agent is not None and (
                    infer_period <= 0.0 or elapsed + 1e-12 >= next_infer_elapsed
                )

                if should_infer:
                    self.policy_last_action = self._infer_policy_action(data, i)
                    if infer_period > 0.0:
                        while next_infer_elapsed <= elapsed + 1e-12:
                            next_infer_elapsed += infer_period

                if self.agent is not None:
                    if self.policy_last_action is None:
                        raise RuntimeError("Policy action state is not initialized")
                    current_ctrl = self._compose_policy_targets(i, self.policy_last_action)
                else:
                    current_ctrl = self.q_targets[i]

                data.ctrl[:] = current_ctrl
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
    parser.add_argument("--agent_path", type=str, default=None, help="Path to ONNX policy. If provided, controls come from policy inference.")
    parser.add_argument("--infer_frequency", type=float, default=50, help="Policy inference frequency in Hz (0 means every step).")
    parser.add_argument("--joint_obs_delay_ms", type=float, default=0, help="Delay applied to joint observations for policy inference (ms).")
    parser.add_argument("--imu_obs_delay_ms", type=float, default=0, help="Delay applied to IMU/gyro observations for policy inference (ms).")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    log = Log(
        str(Path(args.log_path)),
        model=model,
        tracked_joints=[
            "Shoulder",
            "Elbow",
        ],
        dt=args.dt,
        agent_path=args.agent_path,
        infer_frequency=args.infer_frequency,
        joint_obs_delay_ms=args.joint_obs_delay_ms,
        imu_obs_delay_ms=args.imu_obs_delay_ms,
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