import numpy as np
from trajectory import TrajectoryGenerator


class SquatTrajectoryGenerator(TrajectoryGenerator):
    """Generates squat trajectory with configurable parameters.

    Parameters
    ----------
    model_path:      Path to the robot MJCF model.
    leg_spread:      Lateral distance between the two feet (metres). Default 0.2.
    trunk_pitch:     Forward pitch of the trunk (radians). Default 0.0.
    foot_yaw:        Yaw angle of both feet (radians, positive = toes out). Default 0.0.
    squat_amplitude: Half-amplitude of the COM vertical oscillation (metres). Default 0.05.
    frequency:       Squat frequency (Hz). Default 0.5.
    duration:        Total motion duration (seconds). Default 20.0.
    dt:              Control time step (seconds). Default 0.005.
    """

    def __init__(
        self,
        model_path: str,
        leg_spread: float = 0.2,
        trunk_pitch: float = 0.0,
        foot_yaw: float = 0.0,
        squat_amplitude: float = 0.05,
        frequency: float = 0.5,
        duration: float = 20.0,
        dt: float = 0.005,
    ):
        self.leg_spread = leg_spread
        self.trunk_pitch = trunk_pitch
        self.foot_yaw = foot_yaw
        self.squat_amplitude = squat_amplitude
        self.frequency = frequency
        self.duration = duration
        super().__init__(model_path, dt)

    def _setup_tasks(self) -> None:
        # Foot frame tasks (symmetric yaw: left toes out = +yaw, right toes out = -yaw)
        cy, sy = np.cos(self.foot_yaw), np.sin(self.foot_yaw)
        R_yaw_left = np.array([
            [ cy, -sy, 0],
            [ sy,  cy, 0],
            [  0,   0, 1],
        ])
        R_yaw_right = R_yaw_left.T  # opposite yaw for right foot
        T_left_foot = np.eye(4)
        T_left_foot[:3, :3] = R_yaw_left
        self.left_foot_task = self.solver.add_frame_task("left_foot", T_left_foot)
        self.left_foot_task.configure("left_foot_task", "hard", 1.0)

        T_right_foot = np.eye(4)
        T_right_foot[:3, :3] = R_yaw_right
        T_right_foot[1, 3] = -self.leg_spread
        self.right_foot_task = self.solver.add_frame_task("right_foot", T_right_foot)
        self.right_foot_task.configure("right_foot_task", "hard", 1.0)

        # COM position task
        self.p_com_initial = np.array([0.0, -self.leg_spread / 2.0, 0.45])
        self.com_task = self.solver.add_com_task(self.p_com_initial)
        self.com_task.configure("com_task", "soft", 100.0)

        # Trunk orientation task
        cp, sp = np.cos(self.trunk_pitch), np.sin(self.trunk_pitch)
        R_trunk = np.array([
            [ cp, 0, sp],
            [  0, 1,  0],
            [-sp, 0, cp],
        ])
        self.trunk_task = self.solver.add_orientation_task("Trunk", R_trunk)
        self.trunk_task.configure("trunk_task", "soft", 1.0)

        # Upper-body joints task
        joints_task = self.solver.add_joints_task()
        joints_task.set_joints({
            "Head_Yaw": 0.0,
            "Head_Pitch": 0.0,
            "Left_Shoulder_Pitch": 0.0,
            "Left_Shoulder_Roll": -1.0,
            "Left_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": 0.0,
            "Right_Shoulder_Pitch": 0.0,
            "Right_Shoulder_Roll": 1.0,
            "Right_Elbow_Pitch": 0.0,
            "Right_Elbow_Yaw": 0.0,
        })
        joints_task.configure("joint_task", "soft", 1.0)

        self.solver.add_regularization_task(1e-6)

    def _set_initial_config(self) -> None:
        self.robot.set_joint("Left_Knee_Pitch", 0.5)
        self.robot.set_joint("Right_Knee_Pitch", 0.5)
        self.robot.set_joint("Left_Hip_Pitch", -0.25)
        self.robot.set_joint("Right_Hip_Pitch", -0.25)
        self.robot.set_joint("Left_Ankle_Pitch", -0.25)
        self.robot.set_joint("Right_Ankle_Pitch", -0.25)
        self.robot.set_joint("Left_Shoulder_Roll", -1.0)
        self.robot.set_joint("Right_Shoulder_Roll", 1.0)

        T_world_trunk = np.eye(4)
        T_world_trunk[2, 3] = 0.55
        self.robot.set_T_world_frame("Trunk", T_world_trunk)

    def generate(self) -> list[np.ndarray]:
        self.trajectory = []
        t = 0.0

        while t < self.duration:
            t += self.dt

            p_com = self.p_com_initial + np.array([
                0.0,
                0.0,
                -self.squat_amplitude * (1 - np.cos(2 * np.pi * self.frequency * t)),
            ])
            self.com_task.target_world = p_com

            self.solver.solve(True)
            self.robot.update_kinematics()
            self.trajectory.append(self.robot.state.q.copy())

        return self.trajectory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and play a squat trajectory.")
    parser.add_argument("--leg_spread", type=float, default=0.2, help="Lateral distance between the feet (metres).")
    parser.add_argument("--trunk_pitch", type=float, default=0.0, help="Forward pitch of the trunk (radians).")
    parser.add_argument("--foot_yaw", type=float, default=0.0, help="Yaw angle of both feet (radians, positive = toes out).")
    parser.add_argument("--squat_amplitude", type=float, default=0.05, help="Half-amplitude of the COM vertical oscillation (metres).")
    parser.add_argument("--frequency", type=float, default=0.5, help="Squat frequency (Hz).")
    parser.add_argument("--duration", type=float, default=20.0, help="Total motion duration (seconds).")
    args = parser.parse_args()

    gen = SquatTrajectoryGenerator(
        model_path="../humanoid_model/k1/robot.xml",
        leg_spread=args.leg_spread,
        trunk_pitch=args.trunk_pitch,
        foot_yaw=args.foot_yaw,
        squat_amplitude=args.squat_amplitude,
        frequency=args.frequency,
        duration=args.duration,
    )
    gen.generate()
    gen.play()
