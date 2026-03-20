import argparse
from datetime import datetime
import importlib.util
import json
import socket
import sqlite3
import time
from multiprocessing import Process
from pathlib import Path

import mujoco
import numpy as np
import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState

import simulate
from model_wrapper import Actuator, Body, MujocoModelWrapper, Parameter


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
        # armature=Parameter(0.0339552, 0.0, 1.0),
        # forcerange=Parameter(30.0, 20.0, 45.0),
    )

    hip_pitch_actuator = Actuator(
        name="Hip_Pitch",
        model=model,
        dof_names=["Left_Hip_Pitch", "Right_Hip_Pitch"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        # armature=Parameter(0.0478125, 0.0, 1.0),
        # forcerange=Parameter(25.0, 15.0, 40.0),
    )

    hip_yaw_actuator = Actuator(
        name="Hip_Yaw",
        model=model,
        dof_names=["Left_Hip_Yaw", "Right_Hip_Yaw"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        # armature=Parameter(0.0282528, 0.0, 1.0),
        # forcerange=Parameter(20.0, 15.0, 35.0),
    )

    knee_actuator = Actuator(
        name="Knee",
        model=model,
        dof_names=["Left_Knee_Pitch", "Right_Knee_Pitch"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        # armature=Parameter(0.095625, 0.0, 1.0),
        # forcerange=Parameter(45.0, 30.0, 55.0),
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

    trunk_body = Body(
        name="Trunk",
        model=model,
        com_x_offset=Parameter(0.0, -0.05, 0.05),
        com_y_offset=Parameter(0.0, -0.01, 0.01),
        com_z_offset=Parameter(0.0, -0.08, 0.08),
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
        body=[
            trunk_body,
        ],
    )


def list_log_files(logs_dir: Path) -> list[Path]:
    return sorted(logs_dir.rglob("model.log"))


def load_logs(
    log_paths: list[Path],
    agent_log_paths: list[Path],
    model: mujoco.MjModel,
    tracked_joints: list[str],
    dt: float,
) -> list[simulate.Log]:
    logs: list[simulate.Log] = []
    for log_path in log_paths:
        logs.append(
            simulate.Log(
                str(log_path),
                model=model,
                tracked_joints=tracked_joints,
                dt=dt,
            )
        )
    for log_path in agent_log_paths:
        logs.append(
            simulate.Log(
                str(log_path),
                model=model,
                tracked_joints=tracked_joints,
                dt=dt,
                agent_path="walk.onnx",
                infer_frequency=50,
            )
        )
    return logs


def apply_values(params: dict[str, Parameter], values: dict[str, float]) -> None:
    for name, value in values.items():
        params[name].value = float(value)


def weighted_score(joint_mse: float, trunk_angle_mse: float, trunk_weight_ratio: float) -> float:
    return (1.0 - trunk_weight_ratio) * joint_mse + trunk_weight_ratio * trunk_angle_mse


def prepare_sqlite_storage(storage_url: str) -> None:
    sqlite_prefix = "sqlite:///"
    if not storage_url.startswith(sqlite_prefix):
        return

    db_path = storage_url[len(sqlite_prefix):]
    connection = sqlite3.connect(db_path, timeout=60.0)
    try:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA busy_timeout=60000;")
    finally:
        connection.close()


def evaluate_values(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    wrapper: MujocoModelWrapper,
    params: dict[str, Parameter],
    logs: list[simulate.Log],
    values: dict[str, float],
    trunk_weight_ratio: float,
) -> tuple[float, list[float], float, float]:
    apply_values(params, values)
    wrapper.update_model()

    per_log_scores: list[float] = []
    per_log_joint_mse: list[float] = []
    per_log_trunk_angle_mse: list[float] = []
    for log in logs:
        joint_mse, trunk_angle_mse = log.simulate(model=model, data=data, use_viewer=False)
        per_log_joint_mse.append(float(joint_mse))
        per_log_trunk_angle_mse.append(float(trunk_angle_mse))
        score_mse = weighted_score(
            joint_mse=float(joint_mse),
            trunk_angle_mse=float(trunk_angle_mse),
            trunk_weight_ratio=trunk_weight_ratio,
        )
        per_log_scores.append(float(score_mse))

    return (
        float(np.mean(per_log_scores)),
        per_log_scores,
        float(np.mean(per_log_joint_mse)),
        float(np.mean(per_log_trunk_angle_mse)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit MuJoCo model parameters on a directory of logs.")
    parser.add_argument("--logs", type=str, required=None, help="Directory containing logs (model.log).")
    parser.add_argument("--agent-logs", type=str, default=None, help="Directory containing logs to run with a walk agent.")
    parser.add_argument("--trials", type=int, default=1000000, help="Number of Optuna trials.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep.")
    parser.add_argument("--trunk_weight_ratio", type=float, default=0.0, help="Weight ratio in [0, 1] for trunk_angle_mse in final score. 0: joints only, 1: trunk only.")
    parser.add_argument("--sampler", choices=["cmaes", "tpe", "random"], default="cmaes", help="Optuna sampler.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="mujoco_sysid_fit", help="W&B project name.")
    parser.add_argument("--study-storage", type=str, default="sqlite:///study.db", help="Optuna storage URL (used when --workers > 1).")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name (auto-generated if omitted).")
    parser.add_argument("--model", type=str, default="../humanoid_model/k1/scene.xml", help="Path to MuJoCo scene XML.")
    parser.add_argument("--output", type=str, default="params.json", help="Output JSON file for best parameters.")
    args = parser.parse_args()

    if not (0.0 <= args.trunk_weight_ratio <= 1.0):
        raise ValueError(f"--trunk_weight_ratio must be in [0, 1], got: {args.trunk_weight_ratio}")
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got: {args.workers}")

    log_paths = []
    if args.logs:
        logs_dir = Path(args.logs)
        if logs_dir.exists():
            log_paths = list_log_files(logs_dir)

    agent_log_paths = []
    if args.agent_logs:
        agent_logs_dir = Path(args.agent_logs)
        if agent_logs_dir.exists():
            agent_log_paths = list_log_files(agent_logs_dir)

    print(f"Found {len(log_paths)} logs without agent, {len(agent_log_paths)} logs with agent.")

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    wrapper = build_wrapper(model, data)
    params = wrapper.get_parameters()
    logs = load_logs(
        log_paths,
        agent_log_paths,
        model=model,
        tracked_joints=[
            "Shoulder",
            "Elbow",
            "Hip",
            "Knee",
            "Ankle",
        ],
        dt=args.dt,
    )

    baseline_values = {name: float(parameter.value) for name, parameter in params.items()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=output_path.stem,
            config={
                "hostname": socket.gethostname(),
                "sampler": args.sampler,
                "trials": args.trials,
                "workers": args.workers,
                "dt": args.dt,
                "trunk_weight_ratio": args.trunk_weight_ratio,
                "study_storage": args.study_storage if args.workers > 1 else None,
            },
        )

    if args.sampler == "cmaes":
        if importlib.util.find_spec("cmaes") is None:
            print("[fit] 'cmaes' package not found, fallback to TPE sampler.")
            sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(seed=args.seed)
        else:
            sampler = optuna.samplers.CmaEsSampler(seed=args.seed)
    elif args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    elif args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    if args.workers > 1:
        prepare_sqlite_storage(args.study_storage)
        storage = RDBStorage(
            url=args.study_storage,
            engine_kwargs={"connect_args": {"timeout": 60}},
        )
        study_name = args.study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
    else:
        study_name = args.study_name
        study = optuna.create_study(direction="minimize", sampler=sampler)

    if len(study.trials) == 0 and args.workers == 1:
        study.enqueue_trial(baseline_values)

    def objective(trial: optuna.Trial) -> float:
        values: dict[str, float] = {}
        for name, parameter in params.items():
            values[name] = trial.suggest_float(name, parameter.min, parameter.max)

        mean_score, per_log_scores, mean_joint_mse, mean_trunk_angle_mse = evaluate_values(
            model=model,
            data=data,
            wrapper=wrapper,
            params=params,
            logs=logs,
            values=values,
            trunk_weight_ratio=args.trunk_weight_ratio,
        )

        if args.workers == 1:
            for path, score in zip(log_paths, per_log_scores):
                trial.set_user_attr(f"mse:{path}", float(score))

        if wandb_run is not None:
            wandb.log(
                {
                    "optim/trial_mse": float(mean_score),
                    "optim/trial_rmse": float(np.sqrt(mean_score)),
                    "optim/trial_joint_rmse": float(np.sqrt(mean_joint_mse)),
                    "optim/trial_trunk_angle_rmse": float(np.sqrt(mean_trunk_angle_mse)),
                    "optim/trial_number": int(trial.number),
                }
            )

        print(f"Trial {trial.number}/{args.trials - 1}: mean_mse={mean_score:.8f}")
        return float(mean_score)

    last_log_time = 0.0
    def save_best_snapshot(study_obj: optuna.Study) -> None:
        if study_obj.best_trial is None:
            return
        snapshot = {
            "best_mean_mse": float(study_obj.best_value),
            "best_rmse": float(np.sqrt(study_obj.best_value)),
            "best_parameters": {name: float(value) for name, value in study_obj.best_params.items()},
            "meta": {
                "logs_dir": str(logs_dir),
                "nb_logs": len(log_paths),
                "trials": args.trials,
                "workers": args.workers,
                "seed": args.seed,
                "sampler": args.sampler,
                "dt": args.dt,
                "trunk_weight_ratio": args.trunk_weight_ratio,
                "model": args.model,
                "wandb": args.wandb,
                "study_storage": args.study_storage if args.workers > 1 else None,
                "study_name": study_name,
            },
        }
        with output_path.open("w") as file:
            json.dump(snapshot, file, indent=2)

    def monitor(study_obj: optuna.Study, trial: optuna.Trial) -> None:
        nonlocal last_log_time, wandb_run
        now = time.time()
        if now - last_log_time < 0.2:
            return
        last_log_time = now

        if study_obj.best_trial is None:
            return

        save_best_snapshot(study_obj)

        print(f"[Trial {trial.number}] best_mse={study_obj.best_value:.8f}")

        if wandb_run is not None:
            wandb_log = {
                "optim/best_value": float(study_obj.best_value),
                "optim/trial_number": int(trial.number),
            }
            for key, value in study_obj.best_params.items():
                if isinstance(value, float):
                    wandb_log[f"params/{key}"] = float(value)

            wandb.log(wandb_log)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    max_trials_callback = optuna.study.MaxTrialsCallback(args.trials, states=(TrialState.COMPLETE,))

    def run_worker(enable_monitoring: bool) -> None:
        worker_study = study
        if args.workers > 1:
            worker_storage = RDBStorage(
                url=args.study_storage,
                engine_kwargs={"connect_args": {"timeout": 60}},
            )
            worker_study = optuna.load_study(study_name=study_name, storage=worker_storage)

        callbacks: list = [max_trials_callback]
        if enable_monitoring:
            callbacks.append(monitor)

        while True:
            try:
                worker_study.optimize(objective, n_trials=None, n_jobs=1, callbacks=callbacks)
                break
            except ValueError as error:
                if "Cannot tell a COMPLETE trial." not in str(error):
                    raise
                print("[fit] Detected Optuna trial state race, retrying worker loop.")
                time.sleep(0.1)

    worker_processes: list[Process] = []
    if args.workers > 1:
        for _ in range(args.workers - 1):
            process = Process(target=run_worker, args=(False,))
            process.start()
            worker_processes.append(process)

    run_worker(True)

    for process in worker_processes:
        process.join()

    if args.workers > 1:
        final_storage = RDBStorage(
            url=args.study_storage,
            engine_kwargs={"connect_args": {"timeout": 60}},
        )
        study = optuna.load_study(study_name=study_name, storage=final_storage)

    save_best_snapshot(study)

    best_values = {name: float(value) for name, value in study.best_params.items()}
    best_score = float(study.best_value)

    best_score, best_per_log, best_joint_mse, best_trunk_angle_mse = evaluate_values(
        model=model,
        data=data,
        wrapper=wrapper,
        params=params,
        logs=logs,
        values=best_values,
        trunk_weight_ratio=args.trunk_weight_ratio,
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
            "workers": args.workers,
            "seed": args.seed,
            "sampler": args.sampler,
            "dt": args.dt,
            "trunk_weight_ratio": args.trunk_weight_ratio,
            "model": args.model,
            "wandb": args.wandb,
            "study_storage": args.study_storage if args.workers > 1 else None,
            "study_name": study_name,
        },
    }

    with output_path.open("w") as file:
        json.dump(output, file, indent=2)

    if wandb_run is not None:
        wandb.log(
            {
                "optim/best_value": float(best_score),
                "optim/best_rmse": float(np.sqrt(best_score)),
                "optim/best_joint_rmse": float(np.sqrt(best_joint_mse)),
                "optim/best_trunk_angle_rmse": float(np.sqrt(best_trunk_angle_mse)),
            }
        )
        wandb_run.finish()

    print(f"Best mean_mse={best_score:.8f}, rmse={np.sqrt(best_score):.8f}")
    print(f"Saved best parameters to {output_path}")


if __name__ == "__main__":
    main()
