"""Validate a preserved notebook archive without executing or rewriting it."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import warnings

from IPython.core.inputtransformer2 import TransformerManager
import nbformat

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERNS = {
    "github_token": r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    "service_token": r"(?:sk-[A-Za-z0-9_-]{25,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}


def code_text(notebook: dict) -> str:
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")


def code_ast(notebook: dict) -> list[str]:
    transformer = TransformerManager()
    return [ast.dump(ast.parse(transformer.transform_cell("".join(c["source"]))),
                     include_attributes=False)
            for c in notebook["cells"] if c["cell_type"] == "code"]


def compile_embedded(tree: ast.AST, label: str) -> int:
    """Check literal compile() payloads, including stages embedded in stages."""
    count = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "compile" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            source = node.args[0].value
            compile(source, label + ":embedded", "exec")
            count += 1 + compile_embedded(ast.parse(source), label)
    return count


def validate_archive(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "manifest/notebooks.json").read_text(encoding="utf-8"))
    expected_paths = {row["path"] for row in manifest["notebooks"]}
    actual_paths = {p.relative_to(root).as_posix() for p in (root / "notebooks").rglob("*.ipynb")}
    if actual_paths != expected_paths:
        raise ValueError(f"Inventory mismatch: missing={expected_paths - actual_paths}, extra={actual_paths - expected_paths}")
    transformer = TransformerManager()
    checked = []
    notebooks = {}
    for row in manifest["notebooks"]:
        path = root / row["path"]
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"File hash mismatch: {row['path']}")
        raw = json.loads(data)
        with warnings.catch_warnings(record=True) as schema_warnings:
            warnings.simplefilter("always")
            # nbformat may normalize an in-memory object; the file is never saved.
            node = nbformat.reads(data.decode("utf-8"), as_version=4)
            nbformat.validate(node)
        source = code_text(raw)
        if hashlib.sha256(source.encode()).hexdigest() != row["code_sha256"]:
            raise ValueError(f"Code hash mismatch: {row['path']}")
        outputs = [o for c in raw["cells"] for o in c.get("outputs", [])]
        if any(o.get("output_type") == "error" for o in outputs):
            raise ValueError(f"Error outputs in {row['path']}")
        if len(outputs) != row["outputs"]:
            raise ValueError(f"Output count mismatch: {row['path']}")
        for kind, pattern in TOKEN_PATTERNS.items():
            if re.search(pattern, data.decode("utf-8")):
                raise ValueError(f"Credential pattern {kind} in {row['path']}; value omitted")
        code_cells = 0
        embedded = 0
        for index, cell in enumerate(raw["cells"]):
            if cell["cell_type"] != "code":
                continue
            transformed = transformer.transform_cell("".join(cell["source"]))
            label = f"{row['path']}:cell-{index}"
            compile(transformed, label, "exec")
            embedded += compile_embedded(ast.parse(transformed), label)
            code_cells += 1
        notebooks[row["path"]] = raw
        checked.append({"path": row["path"], "sha256": digest, "code_cells": code_cells,
                        "embedded_modules_or_stages": embedded, "outputs": len(outputs),
                        "schema_warnings": len(schema_warnings)})

    pairs = []
    for name in ["08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb", "09_cremad_figuras_cientificas_v3_colab.ipynb"]:
        left, right = "notebooks/" + name, "notebooks/executed/" + name
        if code_text(notebooks[left]) != code_text(notebooks[right]):
            raise ValueError(f"Executed/source mismatch: {name}")
        pairs.append({"source": left, "executed": right, "code_identical": True})
    canonical = root / "notebooks/executed/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb"
    if hashlib.sha256(canonical.read_bytes()).hexdigest() != manifest["canonical_executed_notebook_sha256"]:
        raise ValueError("Canonical thesis hash does not match")
    terminal_text = "\n".join("".join(o.get("text", []))
                              for cell in notebooks[canonical.relative_to(root).as_posix()]["cells"]
                              for o in cell.get("outputs", []))
    if f"EXECUÇÃO CONCLUÍDA E PERSISTIDA: {manifest['canonical_run_id']}" not in terminal_text:
        raise ValueError("Canonical completion marker missing")
    translation_checks = []
    for pt, en in [
        ("08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb", "cremad_emot_definitive_colab.ipynb"),
        ("09_cremad_figuras_cientificas_v3_colab.ipynb", "generate_paper_figures.ipynb"),
    ]:
        equivalent = code_ast(notebooks["notebooks/" + pt]) == code_ast(notebooks["notebooks/publication_en/" + en])
        translation_checks.append({"source": pt, "translation": en, "python_ast_identical": equivalent})
        if not equivalent:
            raise ValueError(f"English code differs beyond comments: {en}")
    sums = {line.split("  ", 1)[1]: line.split("  ", 1)[0]
            for line in (root / "SHA256SUMS").read_text().splitlines() if line}
    if sums != {row["path"]: row["sha256"] for row in manifest["notebooks"]}:
        raise ValueError("SHA256SUMS differs from notebook manifest")
    for directory in [root / "src", root / "scripts", root / "tests"]:
        for path in directory.rglob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return {"status": "passed", "notebooks": len(checked),
            "code_cells": sum(row["code_cells"] for row in checked),
            "embedded_modules_or_stages": sum(row["embedded_modules_or_stages"] for row in checked),
            "error_outputs": 0, "canonical_run_id": manifest["canonical_run_id"],
            "canonical_executed_notebook_sha256": manifest["canonical_executed_notebook_sha256"],
            "source_executed_pairs": pairs, "english_translation_checks": translation_checks,
            "full_experiment_rerun": False, "files": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optionally write the validation report as JSON")
    args = parser.parse_args()
    report = validate_archive()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False, indent=2))
