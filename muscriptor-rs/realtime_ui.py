#!/usr/bin/env python3
"""Live piano-roll UI for `muscriptor-rs --mic --json`.

Spawns the realtime binary, serves a small web page, and forwards each detected
note to the browser over Server-Sent Events. Pure stdlib — no pip install.

    python3 realtime_ui.py                       # auto-detect the release binary
    python3 realtime_ui.py --model small --port 8770
    python3 realtime_ui.py --bin path/to/muscriptor-rs

Then open the printed http://127.0.0.1:<port>/ in a browser.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

subscribers: "set[queue.Queue]" = set()
subscribers_lock = threading.Lock()


def broadcast(note: dict) -> None:
    with subscribers_lock:
        dead = []
        for q in subscribers:
            try:
                q.put_nowait(note)
            except queue.Full:
                dead.append(q)
        for q in dead:
            subscribers.discard(q)


def reader_thread(proc: subprocess.Popen) -> None:
    """Forward each JSONL line from the binary's stdout to subscribers."""
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            broadcast(json.loads(line))
        except json.JSONDecodeError:
            pass  # ignore any non-JSON noise
    print("[realtime_ui] transcriber stdout closed", file=sys.stderr)


INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>MuScriptor — live</title>
<style>
  :root { color-scheme: dark; }
  html,body { margin:0; height:100%; background:#0b0e14; color:#c8d0e0;
    font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; }
  #bar { padding:6px 10px; display:flex; gap:14px; align-items:center;
    border-bottom:1px solid #1c2230; }
  #dot { width:9px; height:9px; border-radius:50%; background:#e5484d; }
  #dot.live { background:#30a46c; box-shadow:0 0 8px #30a46c; }
  canvas { display:block; width:100vw; height:calc(100vh - 34px); }
  b { color:#e8edf7; } .muted { color:#6b7688; }
</style></head><body>
<div id="bar">
  <span id="dot"></span><b>MuScriptor</b>
  <span class="muted">live transcription</span>
  <span id="stat" class="muted">connecting…</span>
  <span id="count" class="muted"></span>
</div>
<canvas id="c"></canvas>
<script>
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const stat = document.getElementById('stat'), dot = document.getElementById('dot');
const countEl = document.getElementById('count');
const LO = 21, HI = 108;            // piano MIDI range
const WINDOW = 12;                  // seconds visible
let notes = [];                     // {start_time,duration,pitch,instrument,is_drum}
let latest = 0, wallAtLatest = performance.now()/1000;

function resize(){ cv.width = innerWidth*devicePixelRatio; cv.height=(innerHeight-34)*devicePixelRatio;
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0); }
addEventListener('resize', resize); resize();

function hue(s){ let h=0; for(const c of s) h=(h*31+c.charCodeAt(0))>>>0; return h%360; }

function draw(){
  const W = cv.width/devicePixelRatio, H = cv.height/devicePixelRatio;
  const now = latest + (performance.now()/1000 - wallAtLatest);
  const t0 = now - WINDOW;
  ctx.clearRect(0,0,W,H);
  // pitch gridlines at octaves (C notes)
  ctx.strokeStyle='#141a26'; ctx.fillStyle='#5a6478'; ctx.lineWidth=1;
  for(let p=LO; p<=HI; p++){ if(p%12!==0) continue;
    const y = H - (p-LO)/(HI-LO)*H;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
    ctx.fillText('C'+(Math.floor(p/12)-1), 3, y-2);
  }
  const rowH = Math.max(3, H/(HI-LO));
  for(const n of notes){
    const x = (n.start_time - t0)/WINDOW * W;
    const w = Math.max(2, (n.duration||0.05)/WINDOW * W);
    if(x+w < 0 || x > W) continue;
    const y = H - (n.pitch-LO)/(HI-LO)*H;
    ctx.fillStyle = n.is_drum ? '#8b93a7' : `hsl(${hue(n.instrument||'x')} 70% 60%)`;
    ctx.globalAlpha = n.is_drum ? 0.7 : 0.9;
    ctx.fillRect(x, y-rowH/2, w, rowH);
  }
  ctx.globalAlpha=1;
  // playhead
  ctx.strokeStyle='#30a46c'; ctx.beginPath(); ctx.moveTo(W-1,0); ctx.lineTo(W-1,H); ctx.stroke();
  // drop notes older than the window
  if(notes.length>4000) notes = notes.filter(n => n.start_time > t0 - WINDOW);
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

const es = new EventSource('/events');
es.onopen = () => { stat.textContent='listening — play music'; dot.classList.add('live'); };
es.onerror = () => { stat.textContent='disconnected'; dot.classList.remove('live'); };
es.onmessage = (e) => {
  const n = JSON.parse(e.data);
  notes.push(n);
  const end = n.start_time + (n.duration||0);
  if(end > latest){ latest = end; wallAtLatest = performance.now()/1000; }
  countEl.textContent = notes.length + ' notes';
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q: queue.Queue = queue.Queue(maxsize=1000)
            with subscribers_lock:
                subscribers.add(q)
            try:
                while True:
                    note = q.get()
                    payload = f"data: {json.dumps(note)}\n\n".encode()
                    self.wfile.write(payload)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with subscribers_lock:
                    subscribers.discard(q)
        else:
            self.send_error(404)


def default_bin() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ("target/release/muscriptor-rs", "target/debug/muscriptor-rs"):
        cand = os.path.join(here, p)
        if os.path.isfile(cand):
            return cand
    return os.path.join(here, "target/release/muscriptor-rs")


def main():
    ap = argparse.ArgumentParser(description="Live piano-roll UI for muscriptor-rs --mic")
    ap.add_argument("--bin", default=default_bin(), help="path to the muscriptor-rs binary")
    ap.add_argument("--model", default="small", help="model size (small/medium/large)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    args, extra = ap.parse_known_args()

    if not os.path.isfile(args.bin):
        sys.exit(f"binary not found: {args.bin}\nBuild it first (e.g. ./build-mac.sh --realtime "
                 "or ./build-cuda.sh --realtime).")

    cmd = [args.bin, "--mic", "--json", "--model", args.model, *extra]
    print(f"[realtime_ui] launching: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    threading.Thread(target=reader_thread, args=(proc,), daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"\n  ▶ open {url}\n", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
