"""Build and cache fixed-shape YOLOE TensorRT engines for single-class text prompts."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

DEFAULT_PROMPT = "soda can"
TEXT_ENGINE = re.compile(r"yoloe-26n-seg-text-(\d+(?:x\d+)?)\.plan")


def default_prompt(model):
    return DEFAULT_PROMPT if TEXT_ENGINE.fullmatch(model) else None


def prompt_engine(base, prompt, publish_progress):
    if prompt is None or prompt == DEFAULT_PROMPT:
        return base

    import tensorrt as trt
    import torch

    from yolo_trt import build

    root = Path(__file__).resolve().parent
    size = TEXT_ENGINE.fullmatch(base.name).group(1)
    weights = base.with_name("yoloe-26n-seg.pt")
    encoder = base.with_name("mobileclip2_b.ts")
    assets = [weights, encoder, root / "export_detectors.py", root / "yolo_trt.py"]
    dependencies = [root / ".yolo-deps", root / ".yolo-clip"]
    properties = torch.cuda.get_device_properties(0)
    signature = ["1", size, prompt, trt.__version__, torch.version.cuda, properties.name,
                 properties.major, properties.minor,
                 *[(asset.name, asset.stat().st_size, asset.stat().st_mtime_ns) for asset in assets]]
    key = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()
    cache = base.parent / "prompts"
    plan = cache / f"{key}.plan"
    if plan.is_file():
        return plan
    if not all(path.is_dir() for path in dependencies):
        raise RuntimeError("Custom prompts require the local .yolo-deps and .yolo-clip export dependencies")

    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=cache) as temporary:
        folder = Path(temporary)
        for asset in (weights, encoder):
            (folder / asset.name).symlink_to(asset)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "YOLO_AUTOINSTALL": "false"}
        env["PYTHONPATH"] = os.pathsep.join(str(path) for path in dependencies)
        publish_progress("Encoding prompt and exporting YOLOE")
        exported = subprocess.run(
            [sys.executable, str(root / "export_detectors.py"), weights.name,
             "--sizes", size, f"--text={prompt}"],
            cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=90, check=False,
        )
        if exported.returncode:
            output = exported.stdout.strip().splitlines()
            reason = output[-1] if output else f"Exporter exited with code {exported.returncode}"
            raise RuntimeError(f"YOLOE prompt export failed. {reason}")
        publish_progress("Building the TensorRT engine")
        pending_plan = folder / "detector.plan"
        build(folder / f"{weights.stem}-text-{size}.onnx", pending_plan)
        pending_plan.replace(plan)
    return plan
