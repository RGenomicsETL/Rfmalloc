#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Export an OpenSpliceAI PyTorch state dictionary as F32 safetensors."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise SystemExit("checkpoint does not contain a state dictionary")

    tensors = {}
    for name, value in state.items():
        if name.endswith(".num_batches_tracked"):
            continue
        if not isinstance(value, torch.Tensor):
            raise SystemExit(f"state entry {name!r} is not a tensor")
        if value.dtype != torch.float32:
            raise SystemExit(f"state tensor {name!r} is not F32")
        tensors[name] = value.detach().cpu().contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output)


if __name__ == "__main__":
    main()
