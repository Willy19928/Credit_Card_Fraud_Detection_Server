import ast
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPO_ROOT / "requirements-dev.txt"
TORCH_JIT_CVE = "CVE-2025-3000"


def run_audit(args: list[str]) -> None:
    command = [sys.executable, "-m", "pip_audit", "--progress-spinner", "off", *args]
    subprocess.run(command, check=True)


def public_version(version: str) -> str:
    return version.split("+", 1)[0]


def production_python_paths() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.py")
        if "tests" not in path.parts and path.resolve() != Path(__file__).resolve()
    )


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def assert_no_torch_jit_script(paths: Iterable[Path] | None = None) -> None:
    violations = []
    for path in paths or production_python_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        torch_aliases = {"torch"}
        jit_aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch":
                        torch_aliases.add(alias.asname or "torch")
                    elif alias.name == "torch.jit":
                        if alias.asname:
                            jit_aliases.add(alias.asname)
                        else:
                            torch_aliases.add("torch")
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                for alias in node.names:
                    if alias.name == "jit":
                        jit_aliases.add(alias.asname or "jit")
            elif isinstance(node, ast.ImportFrom) and node.module == "torch.jit":
                if any(alias.name == "script" for alias in node.names):
                    violations.append(f"{path}: imports torch.jit.script")

        forbidden_names = {
            *(f"{alias}.jit.script" for alias in torch_aliases),
            *(f"{alias}.script" for alias in jit_aliases),
        }
        for node in ast.walk(tree):
            if dotted_name(node) in forbidden_names:
                violations.append(f"{path}: references torch.jit.script")
    if violations:
        raise RuntimeError(
            f"{TORCH_JIT_CVE} may only be ignored while torch.jit.script is unused:\n"
            + "\n".join(sorted(set(violations)))
        )


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
    assert_no_torch_jit_script()
    with tempfile.TemporaryDirectory() as temp_dir:
        requirement_path = Path(temp_dir) / "torch-public-requirement.txt"
        requirement_path.write_text(f"torch=={normalized}\n", encoding="utf-8")
        run_audit(
            [
                "--no-deps",
                "--ignore-vuln",
                TORCH_JIT_CVE,
                "--requirement",
                str(requirement_path),
            ]
        )
    print(
        "Audited torch against public PyPI version "
        f"{normalized} for resolved build {version}. Ignored {TORCH_JIT_CVE}: "
        "the linked upstream report is a torch.jit.script crash bug, PyTorch's "
        "security policy excludes caller-triggered crashes from security "
        "vulnerabilities, and production code does not use torch.jit.script."
    )


def main() -> None:
    dependencies = audit_requirements()
    torch_version = skipped_torch_version(dependencies)
    if torch_version:
        audit_torch_public_version(torch_version)


if __name__ == "__main__":
    main()
