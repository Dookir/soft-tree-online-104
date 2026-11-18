"""Install a private, wheel-hashed CPU environment. No global packages change."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON_VERSION = "3.10.21"


def platform_key():
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "windows-x86_64"
    if sys.platform.startswith("linux") and machine in ("amd64", "x86_64"):
        return "linux-x86_64"
    if sys.platform == "darwin" and machine in ("arm64", "aarch64"):
        return "macos-arm64"
    raise RuntimeError("This installer supports Windows/Linux x86-64 and macOS arm64.")


def download(url, destination):
    try:
        with urllib.request.urlopen(url, timeout=180) as response:
            destination.write_bytes(response.read())
    except Exception:
        # curl uses the OS certificate store on systems whose Python CA bundle
        # is not configured. Certificate verification remains enabled.
        curl = shutil.which("curl")
        if not curl:
            raise
        subprocess.run([curl, "--fail", "--location", "--silent", "--show-error",
                        "--output", str(destination), url], check=True)


def find_uv(key):
    settings = json.loads((ROOT / "config/uv-wheels.json").read_text(encoding="utf-8"))
    row = settings["platforms"][key]
    directory = ROOT / ".tools"
    directory.mkdir(exist_ok=True)
    executable = directory / ("uv.exe" if key.startswith("windows") else "uv")
    if not executable.exists():
        wheel = directory / "uv-bootstrap.whl"
        download(row["url"], wheel)
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("uv download SHA256 mismatch")
        with zipfile.ZipFile(wheel) as archive:
            matching = [name for name in archive.namelist()
                        if Path(name).name == executable.name and archive.getinfo(name).file_size > 1000000]
            if len(matching) != 1:
                raise RuntimeError("Unexpected uv wheel layout")
            executable.write_bytes(archive.read(matching[0]))
        executable.chmod(0o755)
    version = subprocess.check_output([str(executable), "--version"], text=True).strip()
    if not version.startswith("uv " + settings["version"] + " ") and version != "uv " + settings["version"]:
        raise RuntimeError("Wrong uv version: " + version)
    return executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", type=Path, help="Optional existing uv 0.13.7 executable")
    parser.add_argument("--plan", choices=("windows-x86_64", "linux-x86_64", "macos-arm64"))
    args = parser.parse_args()
    key = args.plan or platform_key()
    torch_lock = ROOT / "config" / ("torch-windows.lock" if key.startswith("windows") else
                                    "torch-linux.lock" if key.startswith("linux") else "torch-macos-arm64.lock")
    if args.plan:
        print(json.dumps(dict(platform=key, python=PYTHON_VERSION,
                              uv=json.loads((ROOT / "config/uv-wheels.json").read_text())["platforms"][key],
                              torch_lock=torch_lock.name, common_lock="requirements-common.lock"), indent=2))
        return
    uv = args.uv.resolve() if args.uv else find_uv(key)
    env = os.environ.copy()
    env["UV_PYTHON_INSTALL_DIR"] = str(ROOT / ".tools/python")
    env["UV_CACHE_DIR"] = str(ROOT / ".tools/cache")
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    def run(arguments):
        subprocess.run([str(x) for x in arguments], env=env, check=True)
    if not python.exists():
        run([uv, "--system-certs", "venv", "--python", PYTHON_VERSION, ROOT / ".venv"])
    run([python, "-c", "import sys; assert sys.version_info[:3] == (3,10,21), sys.version"])
    for lock in (ROOT / "requirements-common.lock", torch_lock):
        run([uv, "--system-certs", "pip", "install", "--python", python, "--no-deps",
             "--require-hashes", "--only-binary", ":all:", "--index-url", "https://pypi.org/simple",
             "--requirement", lock])
    run([python, ROOT / "run.py", "doctor"])
    run([python, ROOT / "run.py", "test"])
    print("Setup complete. Training and optimization can now run without network access.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Setup failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
