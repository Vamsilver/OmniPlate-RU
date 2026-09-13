# 🧭 OmniPlate-RU: Agent Documentation Router & Knowledge Index

> **AI AGENT INSTRUCTION**: Read this router FIRST. Do NOT read all sub-documents at once. 
> Locate your current task in the table below, inspect the inlined constants, and ONLY read the specific linked document if deeper context is required.

---

## ⚡ Global Project Invariants (Read Once)
```yaml
TASK: "Recognition of Non-Standard Russian License Plates (Volga IT 2026 / AIS Gorod)"
TARGET_CLASSES: [type1, type1a, type1b, other]
MAX_LATENCY_PER_IMAGE: 100 ms
REFERENCE_HARDWARE: "Intel Core i5-7600, NVIDIA GTX 1050 Ti (4GB VRAM), 16GB RAM"
NETWORK_ACCESS: "STRICTLY FORBIDDEN AT RUNTIME (100% Offline, all weights local)"
DATASET_LICENSE: "CC BY 4.0 (Mandatory)"
CODE_LICENSE: "MIT"
ALLOWED_LETTERS: "ABEKMHOPCTYX" # 12 Latin letters matching Cyrillic
REGEX_PLATE: "^[ABEKMHOPCTYX]\\d{3}[ABEKMHOPCTYX]{2}\\d{2,3}$"
CSV_DELIMITER: ";"
CSV_HEADER: "image;plate_num;plate_type;confidence"
```

---

## 🗺️ Documentation Routing Table

| File | Topic & Key Inlined Constants | Read Condition (When to open) |
|---|---|---|
| [MASTER_PLAN.md](../MASTER_PLAN.md) | **Global Project Roadmap**: Checklists, 5 stages, subtasks, criteria, current progress. | Open to verify next steps or update task completion status. |
| [CONSTITUTION.md](../CONSTITUTION.md) | **Inviolable Rules**: Zero-score conditions, offline ban, CC BY 4.0, face blur, test set isolation. | Open before submission or when validating project constraints. |
| [specs/01_gost_geometry.md](specs/01_gost_geometry.md) | **GOST Dimensions**: <br>• `type1`: 520×112 mm (4.64:1)<br>• `type1a`: 290×170 mm (1.70:1, 2-line)<br>• `type1b`: 520×112 mm, Yellow `#FFCC00`<br>• Font: GOST 50577-2018 | Open ONLY when implementing synthetic plate renderer or homography canonical sizes. |
| [specs/02_plate_mask_regex.md](specs/02_plate_mask_regex.md) | **Plate Mask & Parsing**: <br>• Mask: `[L][D][D][D][L][L][REGION]`<br>• Region: 2 or 3 digits (3-digit starts with 1, 2, 7)<br>• Unreadable char: `#`<br>• Output: Uppercase Latin | Open ONLY when writing OCR post-processor, regex validator, or CTC decoder. |
| [specs/03_hardware_latency_budget.md](specs/03_hardware_latency_budget.md) | **SLA & Speed Budget**: <br>• Target: $\le 100$ ms on GTX 1050 Ti<br>• Detection: $\le 40$ ms<br>• Warp: $\le 5$ ms<br>• OCR: $\le 20$ ms<br>• Memory: $<3.5$ GB VRAM | Open ONLY when benchmarking, profiling, or exporting models to ONNX/TensorRT. |
| [specs/04_dataset_quotas_schema.md](specs/04_dataset_quotas_schema.md) | **Dataset Quotas & Schema**: <br>• Real: 1a $\ge 150$ (50 uniq), 1b $\ge 300$ (100 uniq), other $\ge 50$<br>• Synth: $\ge 5000$<br>• `meta.csv`: 10 columns separated by `;` | Open ONLY when assembling dataset, writing data crawler, or running `validate_dataset.py`. |
| [specs/05_evaluation_metrics.md](specs/05_evaluation_metrics.md) | **Evaluation Math**: <br>• Exact Match (Sequence Acc)<br>• Character Error Rate (CER)<br>• Penalties: predicting `type1` on `other` = fatal | Open ONLY when writing eval scripts or computing training validation loss. |
| [specs/06_extensibility_design.md](specs/06_extensibility_design.md) | **Architecture Extensibility**: <br>• Modularity for CIS plates (BY, KZ, AM)<br>• Motorcycles, military, diplomatic support | Open ONLY when writing the Explanatory Note section on architectural extensibility. |
| [report/01_explanatory_note_draft.md](report/01_explanatory_note_draft.md) | **Submission Report**: 5-page template covering architecture, data, benchmarks, limitations. | Open ONLY when compiling the final report for the jury. |
