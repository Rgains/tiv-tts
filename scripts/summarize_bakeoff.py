#!/usr/bin/env python3
"""Summarize completed Tiv VITS and Matcha-TTS bounded trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_vits_metrics(root: Path) -> Path:
    candidates = list((root / "vits").glob("*/metrics.json"))
    if not candidates:
        raise FileNotFoundError("No VITS metrics found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def instantaneous(history: list[dict[str, Any]], key: str, index: int) -> float:
    if index == 0:
        return float(history[0][key])
    count = index + 1
    return count * float(history[index][key]) - (count - 1) * float(
        history[index - 1][key]
    )


def summarize(root: Path) -> dict[str, Any]:
    matcha_path = root / "matcha" / "metrics.json"
    vits_path = latest_vits_metrics(root)
    matcha = read_json(matcha_path)
    vits = read_json(vits_path)
    vits_history = vits["history"]
    result = {
        "protocol": {
            "sample_count": 60,
            "train_validation_test": [50, 5, 5],
            "steps_per_model": 50,
            "sample_rate": 22_050,
            "character_input": True,
            "pretrained_weights": False,
        },
        "matcha_tts": {
            "parameters": matcha["parameter_count"],
            "seconds_per_step": matcha["seconds_per_step"],
            "initial_train_loss": matcha["initial_train_loss"],
            "final_train_loss": matcha["final_train_loss"],
            "validation_loss": matcha["validation_loss_mean"],
            "checkpoint_bytes": Path(matcha["checkpoint"]).stat().st_size,
            "sample_duration_seconds": matcha["sample_duration_seconds"],
            "sample_audio": matcha["sample_audio"],
            "checkpoint": matcha["checkpoint"],
            "vocoder": matcha["vocoder"],
            "passed": all(
                [
                    matcha["all_losses_finite"],
                    matcha["checkpoint_reload"],
                    Path(matcha["sample_audio"]).is_file(),
                ]
            ),
        },
        "vits": {
            "parameters_including_training_discriminator": vits[
                "parameter_count"
            ],
            "seconds_per_step_including_epoch_validation_overhead": vits[
                "seconds_per_step"
            ],
            "initial_discriminator_loss": instantaneous(
                vits_history, "avg_loss_0", 0
            ),
            "final_discriminator_loss": instantaneous(
                vits_history, "avg_loss_0", len(vits_history) - 1
            ),
            "initial_generator_loss": instantaneous(
                vits_history, "avg_loss_1", 0
            ),
            "final_generator_loss": instantaneous(
                vits_history, "avg_loss_1", len(vits_history) - 1
            ),
            "validation_discriminator_loss": vits["validation_averages"][
                "avg_loss_0"
            ],
            "validation_generator_loss": vits["validation_averages"][
                "avg_loss_1"
            ],
            "checkpoint_bytes": Path(vits["checkpoint"]).stat().st_size,
            "sample_duration_seconds": vits["sample_duration_seconds"],
            "sample_audio": vits["sample_audio"],
            "checkpoint": vits["checkpoint"],
            "vocoder": vits["vocoder"],
            "passed": all(
                [
                    vits["all_losses_finite"],
                    vits["checkpoint_reload"],
                    Path(vits["sample_audio"]).is_file(),
                ]
            ),
        },
        "engineering_comparison": {
            "matcha_parameter_ratio_vs_vits_training_stack": matcha[
                "parameter_count"
            ]
            / vits["parameter_count"],
            "matcha_step_time_ratio_vs_vits": matcha["seconds_per_step"]
            / vits["seconds_per_step"],
            "loss_values_directly_comparable": False,
            "audio_naturalness_ranked": False,
        },
        "recommendation": {
            "aws_primary": "VITS",
            "reason": (
                "Both candidates pass the local gate. VITS is the safer first "
                "end-to-end AWS pilot because one model produces waveform audio "
                "and avoids a separate vocoder selection/licensing/training track."
            ),
            "cost_focused_alternative": "Matcha-TTS plus a separately validated neural vocoder",
            "quality_claim": (
                "No quality winner is claimed from 50 scratch steps. A longer "
                "fixed-budget cloud pilot and blinded Tiv-speaker listening test "
                "are required."
            ),
        },
        "raw_dataset_modified": False,
        "source_metrics": {
            "matcha_tts": str(matcha_path.resolve()),
            "vits": str(vits_path.resolve()),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/bakeoff")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/bakeoff/comparison.json"),
    )
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
