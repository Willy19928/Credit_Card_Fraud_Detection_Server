import importlib.util
from pathlib import Path

import pytest


AUDIT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_dependencies.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("audit_dependencies", AUDIT_PATH)
audit_dependencies = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(audit_dependencies)


def test_torch_jit_cve_exception_accepts_production_code_without_jit(tmp_path):
    safe_path = tmp_path / "safe.py"
    safe_path.write_text("import torch\nmodel = torch.nn.Linear(1, 1)\n", encoding="utf-8")

    audit_dependencies.assert_no_torch_jit_script([safe_path])


@pytest.mark.parametrize(
    "source",
    [
        "import torch\ncompiled = torch.jit.script(lambda value: value)\n",
        "import torch as t\ncompiled = t.jit.script(lambda value: value)\n",
        "import torch.jit as jit\ncompiled = jit.script(lambda value: value)\n",
        "from torch import jit as compiler\ncompiled = compiler.script(lambda value: value)\n",
        "from torch.jit import script\ncompiled = script(lambda value: value)\n",
    ],
)
def test_torch_jit_cve_exception_rejects_jit_script_usage(tmp_path, source):
    unsafe_path = tmp_path / "unsafe.py"
    unsafe_path.write_text(source, encoding="utf-8")

    with pytest.raises(RuntimeError, match="CVE-2025-3000"):
        audit_dependencies.assert_no_torch_jit_script([unsafe_path])
