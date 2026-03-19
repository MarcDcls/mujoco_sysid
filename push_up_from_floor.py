import numpy as np
from trajectory import TrajectoryGenerator


class PushUpFromFloorTrajectoryGenerator(TrajectoryGenerator):
    """Generates a push-up trajectory for the robot lying prone.

    The robot lies face-down with hands and toes flat on the floor.
    Elbows flex and extend to raise and lower the trunk.

    Tasks
    -----
    - Left/right hand position tasks (hard): hands pinned to the ground.
    - Left/right foot position tasks (hard): toes/feet pinned to the ground.
    - Trunk orientation task (soft): trunk kept horizontal.
    - COM position task (soft): drives the vertical oscillation.

    Parameters
    ----------
    model_path:        Path to the robot MJCF model.
    hand_spread:       Lateral distance between the two hands (metres). Default 0.4.
    hand_x_offset:     Forward distance of the hands from the feet (metres). Default 0.5.
    foot_spread:       Lateral distance between the two feet (metres). Default 0.19.
    push_up_height:    COM height above the ground in the extended position (metres). Default 0.30.
    push_up_amplitude: Half-amplitude of the COM vertical oscillation (metres). Default 0.08.
    frequency:         Push-up frequency (Hz). Default 0.5.
    duration:          Total motion duration (seconds). Default 20.0.
    dt:                Control time step (seconds). Default 0.005.
    """

    def __init__(
        self,
        model_path: str,
        hand_spread: float = 0.4,
        hand_x_offset: float = 0.7,
        foot_spread: float = 0.19,
        push_up_amplitude: float = 0.08,
        frequency: float = 0.5,
        duration: float = 20.0,
        dt: float = 0.005,
    ):
        self.hand_spread = hand_spread
        self.hand_x_offset = hand_x_offset
        self.foot_spread = foot_spread
        self.push_up_amplitude = push_up_amplitude
        self.frequency = frequency
        self.duration = duration
        super().__init__(model_path, dt)

    def _setup_tasks(self) -> None:
        # Hand position tasks
        p_left_hand = np.array([self.hand_x_offset, self.hand_spread / 2.0, 0.0])
        self.left_hand_task = self.solver.add_position_task("left_hand", p_left_hand)
        self.left_hand_task.configure("left_hand_task", "soft", 1000)

        p_right_hand = np.array([self.hand_x_offset, -self.hand_spread / 2.0, 0.0])
        self.right_hand_task = self.solver.add_position_task("right_hand", p_right_hand)
        self.right_hand_task.configure("right_hand_task", "soft", 1000)

        # Foot position tasks 
        p_left_foot = np.array([0.0, self.foot_spread / 2.0, 0.0])
        self.left_foot_task = self.solver.add_position_task("left_foot", p_left_foot)
        self.left_foot_task.configure("left_foot_task", "soft", 1000)

        p_right_foot = np.array([0.0, -self.foot_spread / 2.0, 0.0])
        self.right_foot_task = self.solver.add_position_task("right_foot", p_right_foot)
        self.right_foot_task.configure("right_foot_task", "soft", 1000)

        # Head height task
        self.head_height_initial = 0.0
        self.head_task = self.solver.add_position_task("head", np.array([0.0, 0.0, self.head_height_initial]))
        self.head_task.configure("head_height_task", "soft", 100)
        self.head_task.mask.set_axises("z")

        # CoM position task
        self.com_task = self.solver.add_com_task(np.array([0.4, 0.0, self.head_height_initial / 2]))
        self.com_task.configure("com_task", "soft", 100)
        self.com_task.mask.set_axises("y")

        # Joints task
        self.joints_task = self.solver.add_joints_task()
        self.joints_task.set_joints({
            "Head_Yaw": 0.0,
            "Head_Pitch": 0.0,
            "Left_Hip_Roll": 0.0,
            "Right_Hip_Roll": 0.0,
            "Left_Hip_Pitch": 0.0,
            "Right_Hip_Pitch": 0.0,
            "Left_Hip_Yaw": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Left_Knee_Pitch": 0.0,
            "Right_Knee_Pitch": 0.0,
            "Left_Ankle_Roll": 0.0,
            "Right_Ankle_Roll": 0.0,   
            "Left_Ankle_Pitch": 0.0,
            "Right_Ankle_Pitch": 0.0,
        })
        self.joints_task.configure("joints_task", "soft", 1)

        # Joints task
        self.joints_task = self.solver.add_joints_task()
        self.joints_task.set_joints({
            "Left_Shoulder_Roll": -1.0,
            "Right_Shoulder_Roll": 1.0,
            "Left_Shoulder_Pitch": 0.0,
            "Right_Shoulder_Pitch": 0.0,
            "Left_Elbow_Pitch": 0.0,
            "Right_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": 0.0,
            "Right_Elbow_Yaw": 0.0,
        })
        self.joints_task.configure("joints_task", "soft", 1e-2)

        self.solver.add_regularization_task(1e-6)

    def _set_initial_config(self) -> None:
        self.robot.set_joint("Left_Shoulder_Roll", -1.2)
        self.robot.set_joint("Right_Shoulder_Roll", 1.2)
        self.robot.set_joint("Left_Shoulder_Pitch", 1.2)
        self.robot.set_joint("Right_Shoulder_Pitch", 1.2)
        self.robot.set_joint("Left_Elbow_Pitch", -1.3)
        self.robot.set_joint("Right_Elbow_Pitch", -1.3)
        self.robot.set_joint("Left_Elbow_Yaw", -2.2)
        self.robot.set_joint("Right_Elbow_Yaw", 2.2)

        T_world_trunk = np.eye(4)
        T_world_trunk[:3, :3] = np.array([
            [np.cos(-1.5), 0.0, -np.sin(-1.5)],
            [0.0, 1.0, 0.0],
            [np.sin(-1.5), 0.0, np.cos(-1.5)],
        ])
        T_world_trunk[:3, 3] = np.array([1.4, 0.0, -0.4])
        self.robot.set_T_world_frame("Trunk", T_world_trunk)

    def generate(self) -> list[np.ndarray]:
        self.trajectory = []
        t = 0.0
        while t < self.duration:
            t += self.dt

            head_z = self.head_height_initial + self.push_up_amplitude * (1 - np.cos(2 * np.pi * self.frequency * t))
            self.head_task.target_world = np.array([0.0, 0.0, head_z])

            self.solver.solve(True)
            self.robot.update_kinematics()
            self.trajectory.append(self.robot.state.q.copy())

        return self.trajectory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and play a push-up trajectory.")
    parser.add_argument("--hand_spread", type=float, default=0.55, help="Lateral distance between hands (metres).")
    parser.add_argument("--hand_x_offset", type=float, default=0.7, help="Horizontal offset of hands from the center (metres).")
    parser.add_argument("--foot_spread", type=float, default=0.3, help="Lateral distance between feet (metres).")
    parser.add_argument("--push_up_amplitude", type=float, default=0.17, help="Half-amplitude of COM vertical oscillation (metres).")
    parser.add_argument("--frequency", type=float, default=0.5, help="Push-up frequency (Hz).")
    parser.add_argument("--duration", type=float, default=20.0, help="Total motion duration (seconds).")
    args = parser.parse_args()

    gen = PushUpFromFloorTrajectoryGenerator(
        model_path="../humanoid_model/k1/robot.xml",
        hand_spread=args.hand_spread,
        foot_spread=args.foot_spread,
        push_up_amplitude=args.push_up_amplitude,
        frequency=args.frequency,
        duration=args.duration,
        hand_x_offset=args.hand_x_offset,
    )
    gen.generate()
    gen.play()
