#!/usr/bin/env python
"""
Local dashboard for the Image2Splat hot-folder daemon.

Reads daemon.log + scans the hotfolder + datasets/*/probe/ to render a
mobile-friendly HTML page showing live progress: current asset, current
cell (tier/seed), inbox queue, completed count, per-asset cell breakdown,
recent log lines, and a rolling ETA.

USAGE
    python scripts/dashboard.py [--port 8080] [--hotfolder PATH]

ACCESS FROM PHONE
    Same WiFi:
        Find your Windows host IP (`ipconfig` → IPv4 Address on the LAN adapter).
        On phone: open http://<windows-ip>:8080

    Anywhere (recommended): Tailscale
        1. Install Tailscale on Windows (and on phone)
        2. Sign in with same account on both
        3. On phone, open http://<windows-tailscale-name>:8080
        Works on cellular, no port forwarding, end-to-end encrypted.

    One-off public URL: Cloudflare Tunnel
        cloudflared tunnel --url http://localhost:8080
        gives you a temp https URL, no signup required.

Zero non-stdlib deps — pure Python. Drop the script anywhere with the right
--hotfolder pointed at your daemon's working dir.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_HOTFOLDER = Path("/mnt/c/Users/Spenser Dickerson/Desktop/image_to_splat")
DEFAULT_PORT = 8080
LOG_TAIL_LINES = 40
ETA_WINDOW = 5  # rolling average of last N completed assets
EXPECTED_PROBE_CELLS = 10  # fallback when probe_meta.json absent (5 tiers × 2 seeds)


def _safe_dir_list(d: Path, files_only=True) -> list[Path]:
    """List a directory tolerantly — returns [] on missing or permission error."""
    if not d.exists():
        return []
    try:
        entries = list(d.iterdir())
    except (FileNotFoundError, PermissionError):
        return []
    if files_only:
        return [p for p in entries if p.is_file()]
    return entries


def _count_inbox_images(inbox: Path) -> int:
    return sum(1 for p in _safe_dir_list(inbox)
               if p.suffix.lower() in VALID_EXT
               and not p.name.startswith(".")
               and not p.name.endswith(":Zone.Identifier"))


def _read_log_tail(log_path: Path, n: int) -> list[str]:
    if not log_path.exists():
        return []
    try:
        # Tail a large log without loading the whole thing
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 8192
            data = b""
            pos = size
            while pos > 0 and data.count(b"\n") <= n + 2:
                read_size = min(block, pos)
                pos -= read_size
                f.seek(pos)
                data = f.read(read_size) + data
            text = data.decode("utf-8", errors="replace")
        return text.splitlines()[-n:]
    except OSError:
        return []


def _parse_current_state(log_lines: list[str]) -> dict:
    """Scan recent log lines (most-recent-first) for current asset / cell / mode."""
    state = {
        "current_asset": None,
        "current_cell": None,    # e.g. "5/10"
        "current_tier": None,    # e.g. "T3 (Balanced)"
        "current_seed": None,
        "mode": None,            # "probe" | "single-tier" | "batch"
        "last_event_ts": None,
    }
    for line in reversed(log_lines):
        # Most-recent cell line (in probe mode): "  [N/M] tier K (Name) seed=S"
        if state["current_cell"] is None:
            m = re.search(r"\[(\d+)/(\d+)\]\s+tier\s+(\d+)\s+\((\w+)\)\s+seed=(\d+)", line)
            if m:
                state["current_cell"] = f"{m.group(1)}/{m.group(2)}"
                state["current_tier"] = f"T{m.group(3)} ({m.group(4)})"
                state["current_seed"] = m.group(5)
                state["mode"] = "probe"
        # Or batch-mode per-image line: "  [J.I/N] filename.png"
        if state["current_cell"] is None:
            m = re.search(r"\[(\d+)\.(\d+)/(\d+)\]\s+(\S+\.\w+)\s*$", line)
            if m:
                state["current_cell"] = f"{m.group(2)}/{m.group(3)}"
                state["mode"] = "batch"
        # Most-recent INFER or PROBE start
        if state["current_asset"] is None:
            m = re.search(r"(?:INFER|PROBE)\s+(\S+\.\w+)\s+→", line)
            if m:
                state["current_asset"] = m.group(1)
        # Most-recent timestamp
        if state["last_event_ts"] is None:
            m = re.match(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\]", line)
            if m:
                state["last_event_ts"] = m.group(1)
        if (state["current_asset"] and state["current_cell"] and state["last_event_ts"]):
            break
    return state


def _scan_probe_assets(datasets: Path) -> list[dict]:
    """For each slug/probe/ folder, count cells written + total expected."""
    out = []
    if not datasets.exists():
        return out
    for slug_dir in sorted(_safe_dir_list(datasets, files_only=False)):
        if not slug_dir.is_dir():
            continue
        probe_dir = slug_dir / "probe"
        if not probe_dir.is_dir():
            continue
        cells = [p for p in _safe_dir_list(probe_dir)
                 if p.suffix.lower() == ".png" and p.name.startswith("129_T")]
        # probe_meta.json only gets written at the END of a probe — during
        # processing, fall back to EXPECTED_PROBE_CELLS so in-progress assets
        # show "X/10" instead of misleading "X/X".
        expected = EXPECTED_PROBE_CELLS
        meta_path = probe_dir / "probe_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                expected = int(meta.get("batch", {}).get("n_cells_total", expected))
            except (OSError, ValueError, KeyError):
                pass
        # mtime for sorting (most-recent first)
        try:
            mtime = probe_dir.stat().st_mtime
        except OSError:
            mtime = 0
        out.append({
            "slug": slug_dir.name,
            "cells": len(cells),
            "expected": expected,
            "mtime": mtime,
            "done": len(cells) >= expected,
        })
    out.sort(key=lambda x: -x["mtime"])
    return out


def _compute_eta(log_lines: list[str], inbox_count: int) -> dict:
    """Rolling avg time-per-asset from recent DONE lines, project full queue."""
    durations = deque(maxlen=ETA_WINDOW)
    for line in log_lines:
        m = re.search(r"DONE\s+\S+\.\w+\s+in\s+([\d.]+)s", line)
        if m:
            durations.append(float(m.group(1)))
    if not durations or inbox_count == 0:
        return {"avg_seconds": None, "eta_hours": None, "samples": len(durations)}
    avg = sum(durations) / len(durations)
    eta_seconds = avg * inbox_count
    return {
        "avg_seconds": avg,
        "eta_hours": eta_seconds / 3600.0,
        "samples": len(durations),
    }


def get_state(hotfolder: Path) -> dict:
    inbox = hotfolder / "inbox"
    processing = hotfolder / "processing"
    completed = hotfolder / "completed"
    failed = hotfolder / "failed"
    datasets = hotfolder / "datasets"
    log = hotfolder / "daemon.log"

    log_tail = _read_log_tail(log, LOG_TAIL_LINES)
    full_log_tail = _read_log_tail(log, 1000)  # for ETA + state parsing
    current = _parse_current_state(full_log_tail)
    inbox_n = _count_inbox_images(inbox)
    eta = _compute_eta(full_log_tail, inbox_n)

    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "hotfolder": str(hotfolder),
        "queue": {
            "inbox": inbox_n,
            "processing": [p.name for p in _safe_dir_list(processing)],
            "completed": len(_safe_dir_list(completed)),
            "failed": len(_safe_dir_list(failed)),
        },
        "current": current,
        "eta": eta,
        "probe_assets": _scan_probe_assets(datasets),
        "log_tail": log_tail,
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<meta http-equiv="refresh" content="15">
<title>Image2Splat</title>
<style>
:root {{ --bg: #0e1218; --card: #1a1f29; --line: #2a313d; --text: #d6dae3; --muted: #8a93a4; --accent: #61a8ff; --ok: #5cdb95; --warn: #f5b942; --err: #ff6b6b; }}
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, "SF Pro Text", Inter, sans-serif; margin: 0; background: var(--bg); color: var(--text); padding: 12px; max-width: 900px; margin: 0 auto; }}
h1 {{ font-size: 1.15em; margin: 0 0 2px; letter-spacing: -0.01em; }}
h2 {{ font-size: 0.95em; margin: 18px 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
small {{ color: var(--muted); font-size: 0.78em; }}
.card {{ background: var(--card); padding: 12px 14px; border-radius: 10px; margin: 8px 0; border: 1px solid var(--line); }}
.now {{ background: linear-gradient(135deg, #1d3b6e 0%, #1a2a4f 100%); }}
.now .label {{ color: #b8d4ff; font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.1em; }}
.now .val {{ font-size: 1.05em; font-weight: 600; margin-top: 2px; word-break: break-word; }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
.stat {{ background: var(--card); padding: 10px; border-radius: 8px; border: 1px solid var(--line); text-align: center; }}
.stat .n {{ font-size: 1.4em; font-weight: 700; color: var(--accent); }}
.stat .lbl {{ font-size: 0.7em; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }}
.stat.err .n {{ color: var(--err); }}
.eta {{ font-size: 0.85em; color: var(--muted); padding: 6px 4px; }}
.eta strong {{ color: var(--text); }}
.assets {{ background: var(--card); border-radius: 10px; border: 1px solid var(--line); overflow: hidden; }}
.asset {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--line); }}
.asset:last-child {{ border-bottom: none; }}
.asset .slug {{ font-weight: 500; }}
.asset .progress {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.85em; }}
.asset.done .progress {{ color: var(--ok); }}
.asset.partial .progress {{ color: var(--warn); }}
pre.log {{ background: #07090d; padding: 10px; border-radius: 8px; border: 1px solid var(--line); font-size: 0.72em; color: #9aa3b4; overflow-x: auto; max-height: 320px; overflow-y: auto; line-height: 1.4; margin: 0; white-space: pre-wrap; word-break: break-all; }}
.footer {{ color: var(--muted); font-size: 0.7em; margin-top: 20px; text-align: center; padding-bottom: 24px; }}
@media (max-width: 480px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
<h1>🎨 Image2Splat <small style="font-weight: normal;">· {ts}</small></h1>
<small>auto-refresh 15s · hotfolder <code style="font-size: 0.9em;">{hotfolder}</code></small>

<div class="card now">
  <div class="label">Now running</div>
  <div class="val">{current_line}</div>
  <div class="label" style="margin-top: 10px;">Last event</div>
  <div class="val" style="font-size: 0.85em; font-weight: normal;">{last_event_ts}</div>
</div>

<div class="stats">
  <div class="stat"><div class="n">{inbox}</div><div class="lbl">📥 Inbox</div></div>
  <div class="stat"><div class="n">{processing_n}</div><div class="lbl">⏳ Processing</div></div>
  <div class="stat"><div class="n">{completed}</div><div class="lbl">✓ Completed</div></div>
  <div class="stat {failed_cls}"><div class="n">{failed}</div><div class="lbl">✗ Failed</div></div>
</div>

<div class="eta">{eta_line}</div>

<h2>Probe Assets ({asset_count})</h2>
<div class="assets">{asset_rows}</div>

<h2>Log Tail</h2>
<pre class="log">{log}</pre>

<div class="footer">image_to_splat dashboard · stdlib only · {host_port}</div>
</body>
</html>
"""


def render(state: dict, host_port: str) -> str:
    cur = state["current"]
    if cur["current_asset"]:
        parts = [f'<strong>{html.escape(cur["current_asset"])}</strong>']
        if cur["current_cell"]:
            parts.append(f'cell {cur["current_cell"]}')
        if cur["current_tier"]:
            parts.append(html.escape(cur["current_tier"]))
        if cur["current_seed"]:
            parts.append(f'seed {cur["current_seed"]}')
        current_line = " · ".join(parts)
    else:
        current_line = "<em style='color: var(--muted)'>idle — waiting for next asset</em>"

    asset_rows = []
    for a in state["probe_assets"]:
        cls = "done" if a["done"] else "partial"
        sym = "✓" if a["done"] else "⏳"
        asset_rows.append(
            f'<div class="asset {cls}">'
            f'<span class="slug">{html.escape(a["slug"])}</span>'
            f'<span class="progress">{sym} {a["cells"]}/{a["expected"]}</span>'
            f'</div>'
        )
    if not asset_rows:
        asset_rows.append('<div class="asset"><span style="color: var(--muted)"><em>no probe assets yet</em></span></div>')

    eta = state["eta"]
    if eta["avg_seconds"] is None:
        eta_line = f'<em>not enough data yet for ETA (need at least 1 completed asset, have {eta["samples"]})</em>'
    else:
        avg_min = eta["avg_seconds"] / 60.0
        eta_h = eta["eta_hours"]
        finish = datetime.now() + timedelta(hours=eta_h)
        eta_line = (
            f'avg <strong>{avg_min:.1f} min/asset</strong> '
            f'(rolling {eta["samples"]}) · '
            f'queue ETA <strong>{eta_h:.1f}h</strong> · '
            f'finish ≈ <strong>{finish.strftime("%a %H:%M")}</strong>'
        )

    log_text = "\n".join(state["log_tail"])
    return HTML_PAGE.format(
        ts=state["ts"],
        hotfolder=html.escape(state["hotfolder"]),
        current_line=current_line,
        last_event_ts=html.escape(cur["last_event_ts"] or "—"),
        inbox=state["queue"]["inbox"],
        processing_n=len(state["queue"]["processing"]),
        completed=state["queue"]["completed"],
        failed=state["queue"]["failed"],
        failed_cls="err" if state["queue"]["failed"] else "",
        eta_line=eta_line,
        asset_count=len(state["probe_assets"]),
        asset_rows="\n".join(asset_rows),
        log=html.escape(log_text),
        host_port=host_port,
    )


class Handler(BaseHTTPRequestHandler):
    hotfolder: Path
    host_port: str

    def do_GET(self):
        try:
            if self.path == "/" or self.path.startswith("/?"):
                state = get_state(self.hotfolder)
                body = render(state, self.host_port).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                state = get_state(self.hotfolder)
                body = json.dumps(state, indent=2, default=str).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"404")
        except (BrokenPipeError, ConnectionResetError):
            # Client closed connection mid-write; ignore
            pass

    def log_message(self, fmt, *args):
        # Quiet — don't spam stdout on every request
        pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--hotfolder", type=Path, default=DEFAULT_HOTFOLDER,
                   help="Daemon hot-folder root.")
    p.add_argument("--bind", type=str, default="0.0.0.0",
                   help="Bind interface. 0.0.0.0 = all (LAN-accessible). "
                        "127.0.0.1 = localhost only.")
    args = p.parse_args()

    if not args.hotfolder.is_dir():
        print(f"ERROR: hotfolder not a directory: {args.hotfolder}")
        return 1

    Handler.hotfolder = args.hotfolder
    Handler.host_port = f"{args.bind}:{args.port}"
    server = HTTPServer((args.bind, args.port), Handler)
    print("=" * 60)
    print(f"Image2Splat dashboard")
    print(f"  Hotfolder: {args.hotfolder}")
    print(f"  Local:     http://localhost:{args.port}/")
    print(f"  LAN:       http://<your-host-ip>:{args.port}/")
    print(f"  JSON API:  http://localhost:{args.port}/api/state")
    print("=" * 60)
    print("Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
