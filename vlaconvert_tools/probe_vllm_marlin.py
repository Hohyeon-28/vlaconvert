#!/usr/bin/env python3
"""Probe the installed vLLM package for GPTQ-Marlin Linear support."""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback


def main() -> int:
    try:
        import vllm
    except Exception as exc:
        print(f"vLLM import failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"vLLM version: {getattr(vllm, '__version__', '<unknown>')}")
    print(f"vLLM package: {getattr(vllm, '__file__', '<unknown>')}")

    expected = "vllm.model_executor.layers.quantization.gptq_marlin"
    candidates = [expected]
    try:
        discovered = [
            mod.name
            for mod in pkgutil.walk_packages(vllm.__path__, prefix="vllm.")
            if "marlin" in mod.name.lower() and "gptq" in mod.name.lower()
        ]
        candidates.extend(discovered)
    except Exception as exc:
        print(f"Module discovery failed: {type(exc).__name__}: {exc}")

    candidates = list(dict.fromkeys(candidates))
    print("Candidate modules:")
    for name in candidates:
        print(f"  - {name}")

    usable = False
    for name in candidates:
        print(f"\n[{name}]")
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            print(f"IMPORT_ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
            continue

        marlin_exports = sorted(item for item in dir(module) if "Marlin" in item or "marlin" in item)
        print(f"marlin exports: {marlin_exports}")
        has_config = hasattr(module, "GPTQMarlinConfig")
        has_method = hasattr(module, "GPTQMarlinLinearMethod")
        print(f"GPTQMarlinConfig: {has_config}")
        print(f"GPTQMarlinLinearMethod: {has_method}")
        usable = usable or (has_config and has_method)

    if usable:
        print("\nRESULT: GPTQ-Marlin Linear API is available for RealQuant.")
        return 0

    print("\nRESULT: GPTQ-Marlin Linear API was not found in this vLLM install.")
    print("FakeQuant can still run; RealQuant needs a vLLM build exposing GPTQMarlinConfig and GPTQMarlinLinearMethod.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
