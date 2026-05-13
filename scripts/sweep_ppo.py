"""Small sweep runner for PPO hyperparameter experiments.

This runner launches `train_ppo.py` repeatedly with a grid of hyperparameters,
keeps the same seed list across all configurations by default, and writes a
CSV summary plus one log file per run.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


FINAL_UPDATE_RE = re.compile(
    r"update=(?P<update>\d+) step=(?P<step>\d+) "
    r"return=(?P<return>[-+0-9.eE]+) len=(?P<len>[-+0-9.eE]+) "
    r"clip=(?P<clip>[-+0-9.eE]+) kl=(?P<kl>[-+0-9.eE]+)"
)

DIAG_RE = re.compile(
    r"diag policy=(?P<policy>[-+0-9.eE]+) value=(?P<value>[-+0-9.eE]+) "
    r"entropy=(?P<entropy>[-+0-9.eE]+) grad=(?P<grad>[-+0-9.eE]+) "
    r"logit_min=(?P<logit_min>[-+0-9.eE]+) logit_max=(?P<logit_max>[-+0-9.eE]+) "
    r"illegal_rollout=(?P<illegal_rollout>\d+) illegal_update=(?P<illegal_update>\d+)"
)


@dataclass(frozen=True)
class SweepConfig:
    learning_rate: float
    ent_coef: float
    max_grad_norm: float
    dropout: float


def parse_float_list(raw: str) -> List[float]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one value")
    return [float(value) for value in values]


def parse_int_list(raw: str) -> List[int]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one value")
    return [int(value) for value in values]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small PPO hyperparameter sweep")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--train-script", default="train_ppo.py", help="Path to train_ppo.py")
    parser.add_argument("--output-dir", default="sweep_runs", help="Directory for logs and CSV output")
    parser.add_argument("--learning-rates", default="1e-4,3e-4", help="Comma-separated learning rates")
    parser.add_argument("--ent-coefs", default="0.01,0.05", help="Comma-separated entropy coefficients")
    parser.add_argument("--max-grad-norms", default="0.5,1.0", help="Comma-separated grad norm caps")
    parser.add_argument("--dropouts", default="0.0,0.1", help="Comma-separated dropout values")
    parser.add_argument("--seeds", default="123,456,789", help="Comma-separated seeds")
    parser.add_argument("--total-updates", type=int, default=10)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--device", default="auto")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--same-seeds", dest="same_seeds", action="store_true", help="Use the same seed list for every config")
    seed_group.add_argument("--vary-seeds", dest="same_seeds", action="store_false", help="Offset seeds by config index instead of reusing the same list")
    parser.set_defaults(same_seeds=True)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser


def iter_configs(learning_rates: Sequence[float], ent_coefs: Sequence[float], max_grad_norms: Sequence[float], dropouts: Sequence[float]) -> Iterable[SweepConfig]:
    for learning_rate, ent_coef, max_grad_norm, dropout in itertools.product(
        learning_rates, ent_coefs, max_grad_norms, dropouts
    ):
        yield SweepConfig(
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            dropout=dropout,
        )


def build_command(args: argparse.Namespace, config: SweepConfig, seed: int) -> List[str]:
    return [
        args.python,
        args.train_script,
        "--seed",
        str(seed),
        "--total-updates",
        str(args.total_updates),
        "--rollout-length",
        str(args.rollout_length),
        "--num-envs",
        str(args.num_envs),
        "--minibatch-size",
        str(args.minibatch_size),
        "--update-epochs",
        str(args.update_epochs),
        "--learning-rate",
        f"{config.learning_rate:g}",
        "--ent-coef",
        f"{config.ent_coef:g}",
        "--max-grad-norm",
        f"{config.max_grad_norm:g}",
        "--dropout",
        f"{config.dropout:g}",
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--clip-coef",
        str(args.clip_coef),
        "--target-kl",
        str(args.target_kl),
        "--device",
        args.device,
    ]


def parse_run_metrics(stdout: str) -> dict:
    final_update = None
    final_diag = None
    for line in stdout.splitlines():
        update_match = FINAL_UPDATE_RE.search(line)
        if update_match:
            final_update = update_match.groupdict()
            continue
        diag_match = DIAG_RE.search(line)
        if diag_match:
            final_diag = diag_match.groupdict()

    metrics = {
        "update": None,
        "step": None,
        "return": None,
        "len": None,
        "clip": None,
        "kl": None,
        "policy": None,
        "value": None,
        "entropy": None,
        "grad": None,
        "logit_min": None,
        "logit_max": None,
        "illegal_rollout": None,
        "illegal_update": None,
    }

    if final_update is not None:
        metrics.update(final_update)
    if final_diag is not None:
        metrics.update(final_diag)
    return metrics


def main() -> int:
    args = build_arg_parser().parse_args()

    learning_rates = parse_float_list(args.learning_rates)
    ent_coefs = parse_float_list(args.ent_coefs)
    max_grad_norms = parse_float_list(args.max_grad_norms)
    dropouts = parse_float_list(args.dropouts)
    seeds = parse_int_list(args.seeds)

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output_dir / f"ppo_sweep_{timestamp}.csv"

    configs = list(iter_configs(learning_rates, ent_coefs, max_grad_norms, dropouts))
    print(f"Running {len(configs)} configs across {len(seeds)} seed(s). Output: {csv_path}")

    fieldnames = [
        "config_index",
        "seed",
        "learning_rate",
        "ent_coef",
        "max_grad_norm",
        "dropout",
        "exit_code",
        "log_file",
        "update",
        "step",
        "return",
        "len",
        "clip",
        "kl",
        "policy",
        "value",
        "entropy",
        "grad",
        "logit_min",
        "logit_max",
        "illegal_rollout",
        "illegal_update",
        "command",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for config_index, config in enumerate(configs):
            for seed_index, seed in enumerate(seeds):
                effective_seed = seed
                if not args.same_seeds:
                    effective_seed = seed + config_index * 1000

                command = build_command(args, config, effective_seed)
                run_name = (
                    f"cfg{config_index:03d}_seed{effective_seed}_"
                    f"lr{config.learning_rate:g}_ent{config.ent_coef:g}_"
                    f"gn{config.max_grad_norm:g}_do{config.dropout:g}"
                )
                log_file = output_dir / f"{run_name}.log"

                print(f"[{config_index + 1}/{len(configs)} cfg, seed {seed_index + 1}/{len(seeds)}] {run_name}")
                print(" ".join(command))

                if args.dry_run:
                    writer.writerow(
                        {
                            "config_index": config_index,
                            "seed": effective_seed,
                            "learning_rate": config.learning_rate,
                            "ent_coef": config.ent_coef,
                            "max_grad_norm": config.max_grad_norm,
                            "dropout": config.dropout,
                            "exit_code": 0,
                            "log_file": str(log_file),
                            "command": " ".join(command),
                        }
                    )
                    continue

                completed = subprocess.run(
                    command,
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                combined_output = completed.stdout
                if completed.stderr:
                    combined_output += "\n[stderr]\n" + completed.stderr
                log_file.write_text(combined_output, encoding="utf-8")

                metrics = parse_run_metrics(completed.stdout)
                writer.writerow(
                    {
                        "config_index": config_index,
                        "seed": effective_seed,
                        "learning_rate": config.learning_rate,
                        "ent_coef": config.ent_coef,
                        "max_grad_norm": config.max_grad_norm,
                        "dropout": config.dropout,
                        "exit_code": completed.returncode,
                        "log_file": str(log_file),
                        "update": metrics["update"],
                        "step": metrics["step"],
                        "return": metrics["return"],
                        "len": metrics["len"],
                        "clip": metrics["clip"],
                        "kl": metrics["kl"],
                        "policy": metrics["policy"],
                        "value": metrics["value"],
                        "entropy": metrics["entropy"],
                        "grad": metrics["grad"],
                        "logit_min": metrics["logit_min"],
                        "logit_max": metrics["logit_max"],
                        "illegal_rollout": metrics["illegal_rollout"],
                        "illegal_update": metrics["illegal_update"],
                        "command": " ".join(command),
                    }
                )
                csv_file.flush()

                status = "ok" if completed.returncode == 0 else f"failed ({completed.returncode})"
                print(f"  -> {status}; log saved to {log_file}")

    print(f"Sweep complete. Summary CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())