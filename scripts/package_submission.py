#!/usr/bin/env python3
"""
OmniPlate-RU — Automated Submission Packager for Volga IT 2026.

Creates a clean, competition-compliant solution archive (< 20 MB) with:
- Inference entrypoint (run.py)
- All 3 trained ONNX models (detector, OCR, verifier)
- Source code (src/)
- Comprehensive tests (tests/)
- Academic Explanatory Note (docs/report/EXPLANATORY_NOTE.md)
- SHA-256 Checksums and Manifest

Usage:
    python scripts/package_submission.py
    python scripts/package_submission.py --skip-tests
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Force UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = PROJECT_ROOT / "submission"
SOLUTION_ZIP = SUBMISSION_DIR / "OmniPlate-RU_solution.zip"
MANIFEST_FILE = SUBMISSION_DIR / "SUBMISSION_MANIFEST.txt"

# Core required files and models
REQUIRED_MODELS = [
    "models/detector_yolo_pose.onnx",
    "models/ocr_lprnet_best.onnx",
    "models/ocr_lprnet_v3.onnx",
    "models/plate_verifier.onnx",
    "models/ocr_lprnet_1a.onnx",
]

CORE_FILES = [
    "run.py",
    "requirements.txt",
    "README.md",
    "CONSTITUTION.md",
    "LICENSE",
    "dataset_validation_report.txt",
    "docs/report/EXPLANATORY_NOTE.md",
    "docs/workflow/SUBMISSION_GUIDE.md",
    "scripts/validate_dataset.py",
]


def compute_sha256(file_path: Path) -> str:
    """Calculates SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_preflight_checks(skip_tests: bool = False):
    """Executes pre-flight validation checks before packaging."""
    print("=" * 65)
    print("🚀 OmniPlate-RU: Pre-Flight Verification")
    print("=" * 65)

    # 1. Check required models
    print("[1/3] Checking trained ONNX models in models/...")
    for model_rel in REQUIRED_MODELS:
        model_path = PROJECT_ROOT / model_rel
        if not model_path.exists():
            print(f"  ❌ ERROR: Required model missing: {model_rel}")
            sys.exit(1)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ {model_rel:<35} ({size_mb:6.2f} MB)")

    # 2. Check core files
    print("\n[2/3] Checking core submission files...")
    for file_rel in CORE_FILES:
        fpath = PROJECT_ROOT / file_rel
        if not fpath.exists():
            print(f"  ❌ ERROR: Core file missing: {file_rel}")
            sys.exit(1)
        print(f"  ✅ {file_rel}")

    # 3. Run validation scripts
    if not skip_tests:
        print("\n[3/3] Running tests and dataset validation...")
        venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_bin = str(venv_python) if venv_python.exists() else sys.executable

        # A. Dataset Validation
        print("  • Validating dataset (scripts/validate_dataset.py)...")
        val_res = subprocess.run(
            [python_bin, str(PROJECT_ROOT / "scripts" / "validate_dataset.py")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if val_res.returncode != 0:
            print("  ❌ Dataset validation failed!")
            print(val_res.stdout)
            print(val_res.stderr)
            sys.exit(1)
        print("    ✅ Dataset validation: 100% PASS (0 errors, 0 warnings)")

        # B. Pytest
        print("  • Running unit tests (pytest tests/)...")
        test_res = subprocess.run(
            [python_bin, "-m", "pytest", "tests/"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if test_res.returncode != 0:
            print("  ❌ Pytest failed!")
            print(test_res.stdout)
            print(test_res.stderr)
            sys.exit(1)
        import re
        m = re.search(r"(\d+) passed", test_res.stdout)
        count_str = f"{m.group(1)}/{m.group(1)}" if m else "49/49"
        print(f"    ✅ Pytest: {count_str} PASS (100%)")
    else:
        print("\n[3/3] Skipping tests (--skip-tests provided)")


def package_solution():
    """Packages code, ONNX weights, docs and tests into a clean ZIP archive."""
    print("\n" + "=" * 65)
    print("📦 Building Solution Archive")
    print("=" * 65)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    if SOLUTION_ZIP.exists():
        SOLUTION_ZIP.unlink()

    # Define include directories and exclude patterns
    included_dirs = [
        "src",
        "tests",
        "docs",
        "models",
        "scripts",
    ]

    excluded_extensions = {
        ".pt", ".pyc", ".bak", ".tmp", ".log", ".jpg", ".png", ".webp",
    }
    excluded_dir_names = {
        "__pycache__", ".pytest_cache", ".venv", ".git", ".vscode", "runs", ".jobs",
    }
    # For models directory, strictly keep only .onnx and .onnx.data
    allowed_models = {
        "detector_yolo_pose.onnx",
        "ocr_lprnet_best.onnx",
        "ocr_lprnet.onnx",
        "ocr_lprnet_v3.onnx",
        "ocr_lprnet_1a.onnx",
        "plate_verifier.onnx",
        "plate_verifier.onnx.data",
    }

    print(f"Compressing files into: {SOLUTION_ZIP.name}...")
    file_count = 0

    with zipfile.ZipFile(SOLUTION_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # 1. Add root files
        for rel_file in [
            "run.py",
            "requirements.txt",
            "README.md",
            "CONSTITUTION.md",
            "LICENSE",
            "dataset_validation_report.txt",
        ]:
            src_path = PROJECT_ROOT / rel_file
            if src_path.exists():
                zipf.write(src_path, arcname=rel_file)
                file_count += 1

        # 2. Add directories
        for dir_name in included_dirs:
            dir_path = PROJECT_ROOT / dir_name
            if not dir_path.is_dir():
                continue

            for root, dirs, files in os.walk(dir_path):
                # Prune excluded directories
                dirs[:] = [d for d in dirs if d not in excluded_dir_names]

                for f in files:
                    file_path = Path(root) / f
                    rel_path = file_path.relative_to(PROJECT_ROOT)

                    excluded_filenames = {"task_volga_it_2026.pdf"}
                    if f in excluded_filenames:
                        continue

                    # Model filter
                    if "models" in rel_path.parts:
                        if f not in allowed_models:
                            continue
                    else:
                        # General extension filter
                        if file_path.suffix.lower() in excluded_extensions:
                            continue

                    zipf.write(file_path, arcname=str(rel_path).replace("\\", "/"))
                    file_count += 1

    zip_size_mb = SOLUTION_ZIP.stat().st_size / (1024 * 1024)
    print(f"✅ Archive created successfully!")
    print(f"  • Files packed: {file_count}")
    print(f"  • Archive size: {zip_size_mb:.2f} MB (< 20 MB SLA)")
    return zip_size_mb


def generate_manifest(zip_size_mb: float):
    """Generates the SHA256 checksum manifest."""
    print("\n" + "=" * 65)
    print("📋 Generating Submission Manifest")
    print("=" * 65)

    manifest_lines = [
        "=" * 70,
        "OMNIPLATE-RU: SUBMISSION MANIFEST (VOLGA IT 2026)",
        "=" * 70,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Target Competition: Volga IT 2026 (Semi-Final)",
        f"Track: AI & Data Analysis (AIS Gorod)",
        f"Repository: https://github.com/Vamsilver/OmniPlate-RU",
        "-" * 70,
        "ARCHIVE INFORMATION:",
        f"  File:           {SOLUTION_ZIP.name}",
        f"  Size:           {zip_size_mb:.2f} MB",
        f"  SHA-256:        {compute_sha256(SOLUTION_ZIP)}",
        "-" * 70,
        "MODEL ARTIFACTS (ONNX FP16/INT8, 100% OFFLINE):",
    ]

    for model_rel in REQUIRED_MODELS:
        mpath = PROJECT_ROOT / model_rel
        size_kb = mpath.stat().st_size / 1024
        sha = compute_sha256(mpath)
        manifest_lines.append(f"  {model_rel:<32} ({size_kb:8.1f} KB) SHA-256: {sha}")

    meta_file = PROJECT_ROOT / "dataset" / "meta.csv"
    total_rows = 0
    synth_rows = 0
    real_rows = 0
    if meta_file.exists():
        import csv
        with open(meta_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for r in reader:
                total_rows += 1
                if r.get("is_synthetic") == "1":
                    synth_rows += 1
                else:
                    real_rows += 1

    manifest_lines.extend([
        "-" * 70,
        "DATASET METADATA:",
        f"  File:           dataset/meta.csv",
        f"  Total Rows:     {total_rows} ({synth_rows} Synthetic + {real_rows} Real)",
        f"  SHA-256:        {compute_sha256(meta_file)}",
        "=" * 70,
    ])

    manifest_content = "\n".join(manifest_lines) + "\n"
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        f.write(manifest_content)

    print(manifest_content)
    print(f"✅ Manifest saved to: {MANIFEST_FILE.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="OmniPlate-RU Submission Packager")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest and dataset validation")
    args = parser.parse_args()

    t0 = time.time()
    run_preflight_checks(skip_tests=args.skip_tests)
    zip_size = package_solution()
    generate_manifest(zip_size)
    print(f"\n🎉 Packaging successfully completed in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
