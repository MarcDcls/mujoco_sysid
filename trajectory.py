from abc import ABC, abstractmethod
import placo
from placo_utils.visualization import robot_viz, point_viz
import numpy as np
import time


class TrajectoryGenerator(ABC):
    """
    Abstract base class for task-space trajectory generators.
    """

    def __init__(self, model_path: str, dt: float = 0.005):
        self.model_path = model_path
        self.dt = dt
        self.robot = placo.RobotWrapper(model_path, placo.Flags.mjcf)
        self.solver = placo.KinematicsSolver(self.robot)
        self.trajectory: list[np.ndarray] = []

        self._setup_tasks()
        self._warmup()

    @abstractmethod
    def _setup_tasks(self) -> None:
        """Add and configure IK tasks on self.solver."""
        ...

    @abstractmethod
    def generate(self) -> list[np.ndarray]:
        """Run the motion and return the joint-position trajectory."""
        ...

    def _set_initial_config(self) -> None:
        """Pre-set joint positions before the warm-up IK iterations."""
        pass

    def _warmup(self, iterations: int = 20) -> None:
        """Converge the IK to a consistent initial configuration."""
        self._set_initial_config()
        self.robot.update_kinematics()
        for _ in range(iterations):
            self.solver.solve(True)
            self.robot.update_kinematics()

    def play(self) -> None:
        """Replay self.trajectory through the MeshCat viewer at real time."""
        if not self.trajectory:
            raise RuntimeError("No trajectory to display. Call generate() first.")
        
        self.robot.state.q = self.trajectory[0]
        self.robot.update_kinematics()
        viz = robot_viz(self.robot)
        
        # Display at approximately 50 FPS
        skip = max(1, int(0.02 / self.dt))
        i = 0
        for q in self.trajectory:
            step_start = time.perf_counter()

            i += 1
            if i % skip == 0:
                self.robot.state.q = q
                self.robot.update_kinematics()
                viz.display(self.robot.state.q)
                com_proj = self.robot.com_world().copy()
                com_proj[2] = 0.0
                point_viz("com", com_proj, color=0xFF0000, radius=0.01)

            remaining = self.dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)
