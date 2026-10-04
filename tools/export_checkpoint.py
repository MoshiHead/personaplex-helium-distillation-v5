#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a MID-RUN training checkpoint to an inference-ready student file.

    python tools/export_checkpoint.py --checkpoint <run>/best.pt --teacher-checkpoint model.safetensors

`distill/train.py` only exports when a run reaches `--total-steps`, so there is no way to listen to a model
while it is still training. This does exactly what the trainer's end-of-run export does -- build the student
(frozen parts from the teacher checkpoint), load the checkpoint's trainable weights, `export_bf16` -- so the
result drops straight into `moshi.offline --model student --student-checkpoint ...` and the Section 14
knowledge probe.

Runs on CPU by default: it needs ~17 GB of RAM for the teacher weights but no GPU, so it does not disturb a
training job that is using the GPUs. Nothing is written back into the run directory unless you point
--output there.
"""

from pathlib import Path
import argparse
import json
import sys

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "moshi"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch                                                     # noqa: E402
from distill import checkpoint as ckpt                           # noqa: E402
from distill.export import export_bf16                           # noqa: E402
from distill.student_model import build_student_lm               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="best.pt / student_checkpoint.pt / checkpoints/step_*.pt")
    ap.add_argument("--teacher-checkpoint", required=True, help="the teacher's model.safetensors")
    ap.add_argument("--student-config", default=None,
                    help="defaults to the config recorded in the checkpoint's train_args")
    ap.add_argument("--output", default=None, help="defaults to <checkpoint dir>/export_step<N>_bf16.safetensors")
    ap.add_argument("--device", default="cpu", help="cpu (default) keeps the GPUs free for a running job")
    args = ap.parse_args()

    payload = ckpt.load_training_payload(args.checkpoint)
    step = int(payload.get("step", -1))
    train_args = payload.get("train_args") or {}
    cfg_name = args.student_config or train_args.get("student_config")
    if not cfg_name:
        raise SystemExit("--student-config not given and not recorded in the checkpoint")
    selected = [int(x) for x in payload["selected_teacher_layers"]]
    # Phase 4 fine-tunes the depformer; before P4 the checkpoint has no depformer.* keys and inference must
    # keep using the teacher's own copy. Trust the recorded flag, but verify against the state dict.
    has_dep = any(k.startswith("depformer.") for k in payload["student_state_dict"])
    include_dep = bool(payload.get("depformer_trained", False)) and has_dep

    print(f"checkpoint     : {args.checkpoint}")
    print(f"  step         : {step}")
    print(f"  phase        : {(payload.get('schedule_state') or {}).get('phase', '?')}")
    print(f"  best_val     : {(payload.get('extra') or {}).get('best_val')}")
    print(f"  student cfg  : {cfg_name}")
    print(f"  layers       : {selected}")
    print(f"  depformer    : {'included (P4 ran)' if include_dep else 'not included -> teacher copy used'}")

    print(f"\nbuilding student on {args.device} (loads the teacher's frozen parts; ~17 GB RAM on cpu) ...")
    student = build_student_lm(cfg_name, args.teacher_checkpoint, device=args.device, dtype=torch.bfloat16)
    ckpt.load_trainable_state_dict(student, payload["student_state_dict"])

    out = args.output or str(Path(args.checkpoint).parent / f"export_step{step}_bf16.safetensors")
    export_bf16(student, out, selected, include_depformer=include_dep)
    size = Path(out).stat().st_size / 1e9
    print(f"\nexported -> {out}  ({size:.2f} GB)")
    print("\nrun the probe with:")
    print(f"  STUDENT_EXPORT = {out!r}")
    print("  (set it in the Section 14 cell, or pass --student-checkpoint to moshi.offline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
