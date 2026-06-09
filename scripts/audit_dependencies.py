import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPO_ROOT / "requirements-dev.txt"


def run_audit(args: list[str]) -> None:
    command = [sys.executable, "-m", "pip_audit", "--progress-spinner", "off", *args]
    subprocess.run(command, check=True)


def public_version(version: str) -> str:
    return version.split("+", 1)[0]


def audit_requirements() -> list[dict]:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "pip-audit.json"
        run_audit(
            [
                "--format",
                "json",
                "--output",
                str(output_path),
                "--requirement",
                str(REQUIREMENTS_PATH),
            ]
        )
        return json.loads(output_path.read_text(encoding="utf-8"))["dependencies"]


def skipped_torch_version(dependencies: list[dict]) -> str | None:
    for dependency in dependencies:
        if dependency.get("name") != "torch" or "skip_reason" not in dependency:
            continue
        match = re.search(r"\(([^)]+)\)", dependency["skip_reason"])
        if not match:
            raise RuntimeError(
                "pip-audit skipped torch but did not report the skipped version"
            )
        return match.group(1)
    return None


def audit_torch_public_version(version: str) -> None:
    normalized = public_version(version)
    with tempfile.TemporaryDirectory() as temp_dir:
        requirement_path = Path(temp_dir) / "torch-public-requirement.txt"
        requirement_path.write_text(f"torch=={normalized}\n", encoding="utf-8")
        run_audit(["--no-deps", "--requirement", str(requirement_path)])
    print(
        "Audited torch against public PyPI version "
        f"{normalized} for resolved build {version}."
    )


def main() -> None:
    dependencies = audit_requirements()
    torch_version = skipped_torch_version(dependencies)
    if torch_version:
        audit_torch_public_version(torch_version)


if __name__ == "__main__":
    main()
