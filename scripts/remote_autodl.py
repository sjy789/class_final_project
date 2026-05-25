from __future__ import annotations

import argparse
import os
import posixpath
import sys
import tarfile
import time
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]


EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    "dist",
    "outputs",
    "tmp",
    "tools",
    ".venv",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".log",
}


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise SystemExit(f"Missing environment variable: {name}")
    return value


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        env("ML_REMOTE_HOST"),
        port=int(env("ML_REMOTE_PORT", "22")),
        username=env("ML_REMOTE_USER", "root"),
        password=env("ML_REMOTE_PASSWORD"),
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    return client


def run(client: paramiko.SSHClient, command: str, check: bool = True) -> int:
    print(f"$ {command}", flush=True)
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line, end="", flush=True)
    err = stderr.read().decode(errors="replace")
    if err:
        print(err, end="", file=sys.stderr, flush=True)
    code = stdout.channel.recv_exit_status()
    if check and code != 0:
        raise SystemExit(f"remote command failed with exit code {code}")
    return code


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    if path.name.startswith("node-v") and path.suffix == ".zip":
        return True
    if path.name == "OPA.rar":
        return True
    return False


def iter_upload_paths() -> list[Path]:
    include = [
        "backend",
        "training",
        "scripts",
        "docs",
        "samples",
        "datasets/OPA",
        "README.md",
        ".gitignore",
    ]
    paths: list[Path] = []
    for item in include:
        path = ROOT / item
        if path.exists():
            paths.append(path)
    return paths


class ProgressFile:
    def __init__(self, fileobj, total_hint: int | None = None) -> None:
        self.fileobj = fileobj
        self.written = 0
        self.total_hint = total_hint
        self.started = time.time()
        self.last_report = self.started

    def write(self, data: bytes) -> int:
        n = self.fileobj.write(data)
        if n is None:
            n = len(data)
        self.written += n
        now = time.time()
        if now - self.last_report > 10:
            mb = self.written / 1024 / 1024
            rate = mb / max(1e-6, now - self.started)
            print(f"uploaded stream: {mb:.1f} MB at {rate:.1f} MB/s", flush=True)
            self.last_report = now
        return n

    def flush(self) -> None:
        self.fileobj.flush()

    def close(self) -> None:
        self.fileobj.close()


def tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    path = Path(info.name)
    if should_skip(path):
        return None
    return info


def upload(client: paramiko.SSHClient, remote_dir: str) -> None:
    run(client, f"mkdir -p {remote_dir}")
    command = f"tar -xf - -C {remote_dir}"
    channel = client.get_transport().open_session()
    channel.exec_command(command)
    progress = ProgressFile(channel.makefile("wb"))
    with tarfile.open(fileobj=progress, mode="w|") as archive:
        for path in iter_upload_paths():
            arcname = path.relative_to(ROOT).as_posix()
            print(f"adding {arcname}", flush=True)
            archive.add(path, arcname=arcname, filter=tar_filter)
    progress.close()
    code = channel.recv_exit_status()
    err = channel.makefile_stderr("rb").read().decode(errors="replace")
    if err:
        print(err, file=sys.stderr, flush=True)
    if code != 0:
        raise SystemExit(f"remote tar extraction failed with exit code {code}")


def check(client: paramiko.SSHClient, remote_dir: str) -> None:
    cmd = (
        "echo HOST=$(hostname); "
        "pwd; "
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true; "
        "if [ -x /root/miniconda3/bin/python ]; then PYTHON=/root/miniconda3/bin/python; "
        "elif command -v python >/dev/null 2>&1; then PYTHON=$(command -v python); "
        "else PYTHON=$(command -v python3); fi; "
        "echo PYTHON=$PYTHON; "
        "$PYTHON --version; "
        "$PYTHON - <<'PY'\n"
        "try:\n"
        "    import torch\n"
        "    print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n"
        "except Exception as exc:\n"
        "    print('torch import failed:', exc)\n"
        "PY\n"
        "df -h /root /root/autodl-tmp 2>/dev/null || df -h; "
        f"du -sh {remote_dir} 2>/dev/null || true"
    )
    run(client, cmd, check=False)


def remote_setup(client: paramiko.SSHClient, remote_dir: str) -> None:
    command = (
        f"cd {remote_dir} && "
        "if [ -x /root/miniconda3/bin/python ]; then PYTHON=/root/miniconda3/bin/python; "
        "elif command -v python >/dev/null 2>&1; then PYTHON=$(command -v python); "
        "else PYTHON=$(command -v python3); fi && "
        "$PYTHON -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
        "pillow numpy"
    )
    run(client, command)


def train(client: paramiko.SSHClient, remote_dir: str, args: argparse.Namespace) -> None:
    command = (
        f"cd {remote_dir} && "
        "mkdir -p backend/checkpoints && "
        "if [ -x /root/miniconda3/bin/python ]; then PYTHON=/root/miniconda3/bin/python; "
        "elif command -v python >/dev/null 2>&1; then PYTHON=$(command -v python); "
        "else PYTHON=$(command -v python3); fi && "
        "$PYTHON training/train_opa_tiny.py "
        "--data-root datasets/OPA "
        "--output backend/checkpoints/opa_tiny.pt "
        f"--epochs {args.epochs} "
        f"--batch-size {args.batch_size} "
        f"--image-size {args.image_size} "
        f"--num-workers {args.num_workers} "
        "--device cuda"
    )
    if args.max_train_samples:
        command += f" --max-train-samples {args.max_train_samples}"
    if args.max_val_samples:
        command += f" --max-val-samples {args.max_val_samples}"
    run(client, command)


def download_checkpoint(client: paramiko.SSHClient, remote_dir: str) -> None:
    sftp = client.open_sftp()
    local_dir = ROOT / "backend" / "checkpoints"
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in ["opa_tiny.pt", "opa_tiny.history.json"]:
        remote_path = posixpath.join(remote_dir, "backend", "checkpoints", name)
        local_path = local_dir / name
        print(f"downloading {remote_path} -> {local_path}", flush=True)
        sftp.get(remote_path, str(local_path))
    sftp.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "upload", "setup", "train", "download", "all"])
    parser.add_argument("--remote-dir", default="/root/autodl-tmp/machine_learning_project")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = connect()
    try:
        if args.action == "check":
            check(client, args.remote_dir)
        elif args.action == "upload":
            upload(client, args.remote_dir)
        elif args.action == "setup":
            remote_setup(client, args.remote_dir)
        elif args.action == "train":
            train(client, args.remote_dir, args)
        elif args.action == "download":
            download_checkpoint(client, args.remote_dir)
        elif args.action == "all":
            check(client, args.remote_dir)
            upload(client, args.remote_dir)
            remote_setup(client, args.remote_dir)
            train(client, args.remote_dir, args)
            download_checkpoint(client, args.remote_dir)
    finally:
        client.close()


if __name__ == "__main__":
    main()
