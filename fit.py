import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import optuna
import placo

import simulate
from model_wrapper import Actuator, MujocoModelWrapper, Parameter


def build_wrapper(model: mujoco.MjModel, data: mujoco.MjData) -> MujocoModelWrapper:
    arm_actuator = Actuator(
        name="Arm",
        model=model,
        dof_names=[
            "Left_Shoulder_Pitch",
            "Right_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Right_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Right_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "Right_Elbow_Yaw",
        ],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.001, 0.0, 1.0),
        forcerange=Parameter(10.0, 5.0, 15.0),
    )

    hip_roll_actuator = Actuator(
        name="Hip_Roll",
        model=model,
        dof_names=["Left_Hip_Roll", "Right_Hip_Roll"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.0339552, 0.0, 1.0),
        forcerange=Parameter(30.0, 20.0, 45.0),
    )

    hip_pitch_actuator = Actuator(
        name="Hip_Pitch",
        model=model,
        dof_names=["Left_Hip_Pitch", "Right_Hip_Pitch"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.0478125, 0.0, 1.0),
        forcerange=Parameter(25.0, 15.0, 40.0),
    )

    hip_yaw_actuator = Actuator(
        name="Hip_Yaw",
        model=model,
        dof_names=["Left_Hip_Yaw", "Right_Hip_Yaw"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.0282528, 0.0, 1.0),
        forcerange=Parameter(20.0, 15.0, 35.0),
    )

    knee_actuator = Actuator(
        name="Knee",
        model=model,
        dof_names=["Left_Knee_Pitch", "Right_Knee_Pitch"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.095625, 0.0, 1.0),
        forcerange=Parameter(45.0, 30.0, 55.0),
    )

    ankle_roll_actuator = Actuator(
        name="Ankle_Roll",
        model=model,
        dof_names=["Left_Ankle_Roll", "Right_Ankle_Roll"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.0565, 0.0, 1.0),
        forcerange=Parameter(20.0, 5.0, 30.0),
    )

    ankle_pitch_actuator = Actuator(
        name="Ankle_Pitch",
        model=model,
        dof_names=["Left_Ankle_Pitch", "Right_Ankle_Pitch"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.0565, 0.0, 1.0),
        forcerange=Parameter(20.0, 5.0, 30.0),
    )

    return MujocoModelWrapper(
        model=model,
        data=data,
        actuator=[
            arm_actuator,
            hip_roll_actuator,
            hip_pitch_actuator,
            hip_yaw_actuator,
            knee_actuator,
            ankle_roll_actuator,
            ankle_pitch_actuator,
        ],
    )


def list_log_files(logs_dir: Path) -> list[Path]:
    files = sorted(logs_dir.rglob("model.log"))
    if not files:
        files = sorted(logs_dir.rglob("*.log"))
    return files


def load_histories(log_paths: list[Path]) -> list[placo.HistoryCollection]:
    histories: list[placo.HistoryCollection] = []
    for log_path in log_paths:
        history = placo.HistoryCollection()
        history.loadReplays(str(log_path))
        histories.append(history)
    return histories


def apply_values(params: dict[str, Parameter], values: dict[str, float]) -> None:
    for name, value in values.items():
        params[name].value = float(value)


def evaluate_values(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    wrapper: MujocoModelWrapper,
    params: dict[str, Parameter],
    histories: list[placo.HistoryCollection],
    values: dict[str, float],
    dt: float,
    support_foot: str,
) -> tuple[float, list[float]]:
    apply_values(params, values)
    wrapper.update_model()

    per_log_scores: list[float] = []
    for history in histories:
        score_mse, _ = simulate.simulate(
            model=model,
            data=data,
            history=history,
            dt=dt,
            use_viewer=False,
            support_foot=support_foot,
        )
        per_log_scores.append(float(score_mse))

    return float(np.mean(per_log_scores)), per_log_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit MuJoCo model parameters on a directory of logs.")
    parser.add_argument("--logs-dir", type=str, required=True, help="Directory containing logs (model.log).")
    parser.add_argument("--trials", type=int, default=100000, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep.")
    parser.add_argument("--support-foot", choices=["left", "right"], default="left")
    parser.add_argument("--sampler", choices=["cmaes", "tpe"], default="cmaes", help="Optuna sampler.")
    parser.add_argument("--model", type=str, default="../humanoid_model/k1/scene.xml", help="Path to MuJoCo scene XML.")
    parser.add_argument("--output", type=str, default="params.json", help="Output JSON file for best parameters.")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory does not exist: {logs_dir}")

    log_paths = list_log_files(logs_dir)
    if not log_paths:
        raise RuntimeError(f"No log files found under: {logs_dir}")

    print(f"Found {len(log_paths)} logs")
    for path in log_paths:
        print(f"  - {path}")

    histories = load_histories(log_paths)

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    wrapper = build_wrapper(model, data)
    params = wrapper.get_parameters()

    baseline_values = {name: float(parameter.value) for name, parameter in params.items()}

    if args.sampler == "cmaes":
        sampler: optuna.samplers.BaseSampler = optuna.samplers.CmaEsSampler(restart_strategy="bipop")
    elif args.sampler == "random":
        sampler = optuna.samplers.RandomSampler()
    elif args.sampler == "nsgaii":
        sampler = optuna.samplers.NSGAIISampler()
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(baseline_values)

    def objective(trial: optuna.Trial) -> float:
        values: dict[str, float] = {}
        for name, parameter in params.items():
            values[name] = trial.suggest_float(name, parameter.min, parameter.max)

        mean_score, per_log_scores = evaluate_values(
            model=model,
            data=data,
            wrapper=wrapper,
            params=params,
            histories=histories,
            values=values,
            dt=args.dt,
            support_foot=args.support_foot,
        )

        for path, score in zip(log_paths, per_log_scores):
            trial.set_user_attr(f"mse:{path}", float(score))

        print(f"Trial {trial.number}/{args.trials - 1}: mean_mse={mean_score:.8f}")
        return float(mean_score)

    study.optimize(objective, n_trials=args.trials)

    best_values = {name: float(value) for name, value in study.best_params.items()}
    best_score = float(study.best_value)

    best_score, best_per_log = evaluate_values(
        model=model,
        data=data,
        wrapper=wrapper,
        params=params,
        histories=histories,
        values=best_values,
        dt=args.dt,
        support_foot=args.support_foot,
    )

    output = {
        "best_mean_mse": best_score,
        "best_rmse": float(np.sqrt(best_score)),
        "best_per_log_mse": {str(path): score for path, score in zip(log_paths, best_per_log)},
        "best_parameters": best_values,
        "meta": {
            "logs_dir": str(logs_dir),
            "nb_logs": len(log_paths),
            "trials": args.trials,
            "seed": args.seed,
            "sampler": args.sampler,
            "dt": args.dt,
            "support_foot": args.support_foot,
            "model": args.model,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(output, file, indent=2)

    print(f"Best mean_mse={best_score:.8f}, rmse={np.sqrt(best_score):.8f}")
    print(f"Saved best parameters to {output_path}")


if __name__ == "__main__":
    main()
