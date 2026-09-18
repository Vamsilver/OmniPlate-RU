#!/usr/bin/env python3
"""
OmniPlate-RU — Local Live Audit Server.
Serves test_output/gallery.html at http://localhost:8080 and automatically
syncs user audit votes & Ground Truth corrections directly to CSV on disk.
"""

import csv
import http.server
import json
import socketserver
import webbrowser
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = PROJECT_ROOT / "test_output" / "gallery.html"
CSV_PATH = PROJECT_ROOT / "test_output" / "omniplate_ground_truth_audit.csv"
REGISTRY_PATH = PROJECT_ROOT / "test_output" / "audited_registry.json"
PORT = 8080
REVIEW_DIR = None
REVIEW_CSV_PATH = None
LIMIT = 60


def sync_registry(existing_records: dict) -> dict:
    if REVIEW_DIR is not None and REVIEW_DIR.exists():
        all_imgs = [f.name.lower() for f in REVIEW_DIR.rglob("*") if f.suffix.lower() in (".jpg", ".png")]
        total_real = len(all_imgs) if all_imgs else 52
    else:
        img_dir = PROJECT_ROOT / "dataset" / "images" / "real"
        all_imgs = [f.name.lower() for f in img_dir.glob("*") if f.suffix.lower() in (".jpg", ".png")] if img_dir.exists() else []
        total_real = len(all_imgs) if all_imgs else 531

    audited_count = len(existing_records)
    unseen_count = max(0, total_real - audited_count)
    progress_pct = round((audited_count / max(1, total_real)) * 100.0, 1)

    reg_data = {
        "total_real_images": total_real,
        "audited_count": audited_count,
        "unseen_count": unseen_count,
        "progress_percent": progress_pct,
        "audited_filenames": sorted(list(existing_records.keys())),
    }
    try:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(reg_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[REGISTRY WARNING] Could not write registry: {e}")
    return reg_data


class LiveAuditHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        clean_path = self.path.split("?")[0]
        if clean_path in ("/", "/index.html", "/gallery.html"):
            if not HTML_PATH.exists():
                self.send_error(404, "gallery.html not found. Run scripts/build_visual_gallery.py first.")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            with open(HTML_PATH, "rb") as f:
                self.wfile.write(f.read())
            return
        elif clean_path == "/api/status" or clean_path == "/api/stats":
            reg_data = {}
            if REGISTRY_PATH.exists():
                try:
                    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                        reg_data = json.load(f)
                except Exception:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            resp = {"status": "ok", "stats": reg_data}
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return
        super().do_GET()

    def do_POST(self):
        clean_path = self.path.split("?")[0]
        if clean_path == "/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8"))
                saved_cnt, reg_info = self.save_csv(payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                res = {
                    "status": "saved",
                    "path": "test_output/omniplate_ground_truth_audit.csv",
                    "audited_count": saved_cnt,
                    "total_count": reg_info.get("total_real_images", 531),
                    "progress_percent": reg_info.get("progress_percent", 0.0),
                }
                self.wfile.write(json.dumps(res).encode("utf-8"))
                print(f"[LIVE AUDIT] Successfully auto-saved {saved_cnt} votes to {CSV_PATH.name} and registry")
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                err_msg = json.dumps({"status": "error", "message": str(e)})
                self.wfile.write(err_msg.encode("utf-8"))
                print(f"[LIVE AUDIT ERROR] Failed to save: {e}")
            return
        elif clean_path == "/api/next_batch":
            try:
                import subprocess
                venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
                py_cmd = str(venv_py) if venv_py.exists() else sys.executable
                if REVIEW_DIR is not None:
                    cmd = [py_cmd, str(PROJECT_ROOT / "scripts" / "build_visual_gallery.py"), "--input_dir", str(REVIEW_DIR), "--include_audited", "--limit", str(LIMIT)]
                else:
                    cmd = [py_cmd, str(PROJECT_ROOT / "scripts" / "build_visual_gallery.py"), "--shuffle", "--limit", "45"]
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=180)
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr or proc.stdout)

                reg_info = {}
                if REGISTRY_PATH.exists():
                    try:
                        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                            reg_info = json.load(f)
                    except Exception:
                        pass

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                res = {
                    "status": "ok",
                    "audited_count": reg_info.get("audited_count", 0),
                    "total_count": reg_info.get("total_real_images", 531),
                    "unseen_count": reg_info.get("unseen_count", 0),
                }
                self.wfile.write(json.dumps(res).encode("utf-8"))
                print("[LIVE AUDIT] Successfully generated fresh unseen batch gallery!")
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                err_msg = json.dumps({"status": "error", "message": str(e)})
                self.wfile.write(err_msg.encode("utf-8"))
                print(f"[LIVE AUDIT ERROR] Failed to generate next batch: {e}")
            return
        self.send_error(404)

    def save_csv(self, payload):
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing_records = {}
        if CSV_PATH.exists():
            try:
                with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f, delimiter=";")
                    header = next(reader, None)
                    for row in reader:
                        if len(row) >= 6 and row[1]:
                            norm_name = Path(row[1].strip()).name.lower()
                            existing_records[norm_name] = row
            except Exception:
                pass

        if isinstance(payload, list):
            for row in payload:
                if len(row) >= 6 and row[0] != "id":
                    fname = row[1].strip()
                    norm_name = Path(fname).name.lower()
                    status = row[4].strip()
                    if status != "unreviewed":
                        existing_records[norm_name] = row
        elif isinstance(payload, dict):
            for k, v in payload.items():
                fname = v.get("filename", "").strip()
                norm_name = Path(fname).name.lower()
                status = v.get("status", "").strip()
                if fname and status and status != "unreviewed":
                    existing_records[norm_name] = [
                        k,
                        fname,
                        v.get("type", ""),
                        v.get("predicted_text", ""),
                        status,
                        v.get("ground_truth_text", "")
                    ]

        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["id", "filename", "type", "predicted_text", "status", "ground_truth_text"])
            for idx, (norm_name, row) in enumerate(sorted(existing_records.items(), key=lambda x: x[1][1].lower()), start=1):
                ptype = "other" if str(row[2]).strip().lower() == "type2" else row[2]
                writer.writerow([idx, row[1], ptype, row[3], row[4], row[5]])

        cand_dir = PROJECT_ROOT / "dataset" / "candidates_review"
        target_rev_csv = REVIEW_CSV_PATH or (cand_dir / "review_decisions.csv" if cand_dir.exists() else None)
        if target_rev_csv is not None:
            try:
                target_rev_csv.parent.mkdir(parents=True, exist_ok=True)
                cand_records = []
                for norm_name, row in sorted(existing_records.items(), key=lambda x: x[1][1].lower()):
                    fn = row[1].replace("\\", "/")
                    if "type1a/cand_" in fn or "type2/cand_" in fn or "cand_" in fn:
                        ptype = "other" if str(row[2]).strip().lower() == "type2" else row[2]
                        cand_records.append([row[0], row[1], ptype, row[3], row[4], row[5]])
                if cand_records:
                    with open(target_rev_csv, "w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f, delimiter=";")
                        writer.writerow(["id", "filename", "type", "predicted_text", "status", "ground_truth_text"])
                        for idx, r in enumerate(cand_records, start=1):
                            writer.writerow([idx, r[1], r[2], r[3], r[4], r[5]])
            except Exception as e:
                print(f"[REVIEW WARNING] Could not write review decisions: {e}")

        reg_info = sync_registry(existing_records)
        return len(existing_records), reg_info


def run():
    global PORT, REVIEW_DIR, REVIEW_CSV_PATH, LIMIT
    parser = argparse.ArgumentParser(description="OmniPlate Live Audit & Candidate Review Server")
    parser.add_argument("--review-dir", type=str, default=None, help="Staging review directory (e.g. dataset/candidates_review)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--limit", type=int, default=100, help="Max candidates per gallery batch (default: 100)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    PORT = args.port
    LIMIT = args.limit

    if args.review_dir:
        rev_p = Path(args.review_dir)
        REVIEW_DIR = rev_p if rev_p.is_absolute() else (PROJECT_ROOT / rev_p)
        REVIEW_CSV_PATH = REVIEW_DIR / "review_decisions.csv"

        print("=" * 65)
        print(f"[*] Режим: РУЧНАЯ ПРОВЕРКА КАНДИДАТОВ (STAGING REVIEW)")
        print(f"[*] Директория кандидатов: {REVIEW_DIR}")
        print(f"[*] Генерация / обновление gallery.html для кандидатов...")
        print("=" * 65)

        import subprocess
        venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        py_cmd = str(venv_py) if venv_py.exists() else sys.executable
        cmd = [py_cmd, str(PROJECT_ROOT / "scripts" / "build_visual_gallery.py"), "--input_dir", str(REVIEW_DIR), "--include_audited", "--limit", str(LIMIT)]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if proc.returncode != 0:
            print(f"[!] Warning: build_visual_gallery exited with code {proc.returncode}")

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", PORT), LiveAuditHandler) as httpd:
        print("=" * 60)
        print(f"🚀 OmniPlate Live Audit Server запущен на http://localhost:{PORT}")
        if REVIEW_DIR:
            print(f"🎯 Staging кандидаты: {REVIEW_DIR}")
            print(f"📝 Решения ручной проверки сохраняются в:")
            print(f"   {REVIEW_CSV_PATH}")
        print(f"📁 Все правки и клики также сбрасываются в:")
        print(f"   {CSV_PATH}")
        print("=" * 60)
        print("Для выхода нажмите Ctrl+C в этом окне.")
        if not args.no_browser:
            try:
                webbrowser.open(f"http://localhost:{PORT}")
            except Exception:
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[LIVE AUDIT] Сервер остановлен.")


if __name__ == "__main__":
    run()
