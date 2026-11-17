"""Cross-validation scheduler: nine runs across five Kaggle notebooks (cv_p2..4 need GPU,
cv_a2..4 can use CPU, cv_f2..4 depend on their cv_p counterparts). Every two minutes, a free slot
(idle with previous output downloaded) receives the next ready run. When cv_p finishes, pack
models/cv_pK, commit and push nap3/long-exp, then enable cv_fK. Log progress to cv_sched.log.
"""
import json
import os
import subprocess
import time
from pathlib import Path

R = Path("/Users/bunnypro/Projects/more-than-a-hack-2027-nap3-dev")
S = Path("/private/tmp/claude-501/-Users-bunnypro--claude-spares-spare-cVXuLW/9ba5666f-d507-4884-976f-e9ae159151fb/scratchpad")
P = "/private/tmp/claude-501/-Users-bunnypro--claude-spares-spare-1S3e4Z/8ab43240-bfd8-407e-8e87-39c89f91c576/scratchpad/venv-nap2/bin/python"
ENV = dict(os.environ, PYTHONPATH=".")
# 20:51: weekly GPU quota exhausted (30 hours); all slots use CPU from here.
GPU = []
CPU = ["mtah-nap3-train", "mtah-nap3-train-2", "mtah-nap3-train-3", "mtah-nap3-train-4", "mtah-nap3-train-5"]
# (name, requires GPU, dependencies)
JOBS = [("cv_p2", True, None), ("cv_p3", True, None), ("cv_a2", False, None), ("cv_a3", False, None),
        ("cv_a4", False, None), ("cv_f2", False, "cv_p2"), ("cv_f3", False, "cv_p3")]
# 21:15: fold 4 (cv_p4, cv_f4) moved to a separate GPU run (mac_fold4.sh); leave Kaggle cv_p4 running without downloading it.
LOG = S / "cv_sched.log"
slot_job, started, packed, procs = {}, set(), set(), []
# On restart, reconstruct state from existing logs.
import re
for l in (LOG.read_text().splitlines() if LOG.is_file() else []):
    m = re.match(r"\d\d:\d\d launch (\S+) on (\S+?)(?: \(CPU\))?: ok$", l)
    if m:
        slot_job[m.group(2)] = m.group(1); started.add(m.group(1))
    m = re.match(r"\d\d:\d\d pack (\S+):.*push rc 0", l)
    if m:
        packed.add(m.group(1))


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M ") + msg + "\n")


def done(exp):
    return (R / "artifacts" / "exp" / exp / "eval.json").is_file()


def failed(exp):
    b = S / "batch_board.tsv"
    return b.is_file() and any(l.startswith(exp + "\tLỖI") for l in b.read_text().splitlines())


def free(slot):
    j = slot_job.get(slot)
    return j is None or done(j) or failed(j)


def pack(exp):
    sh = lambda *c: subprocess.run(c, cwd=R, env=ENV, capture_output=True, text=True)
    r = sh(P, "-m", "src.submit", "pack", exp)
    sh("git", "add", "models/" + exp)
    c = sh("git", "commit", "-m", "models/%s: CV fold pre-training checkpoint (for %s)\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
           % (exp, exp.replace("cv_p", "cv_f")), "--", "models/" + exp)
    p = sh("git", "push", "-q", "team", "nap3/long-exp")
    log("pack %s: %s | commit %s | push rc %d %s" % (exp, r.stdout.strip()[-80:], c.returncode, p.returncode, p.stderr.strip()[-200:]))
    return p.returncode == 0


def launch(exp, slot, cpu):
    cmd = [P, "-m", "src.kaggle_run", "launch", "--exp", "exps/%s.toml" % exp, "--slot", slot, "--remote", "team"]
    if cpu:
        cmd.append("--cpu")
    r = subprocess.run(cmd, cwd=R, env=ENV, capture_output=True, text=True)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "token" not in l.lower())
    ok = r.returncode == 0 and "successfully pushed" in out
    log("launch %s on %s%s: %s" % (exp, slot, " (CPU)" if cpu else "", "ok" if ok else "FAIL " + out[-300:]))
    if ok:
        slot_job[slot] = exp
        started.add(exp)
        procs.append(subprocess.Popen([str(S / "auto_exp.sh"), exp, slot, "noscore"]))
    return ok


for sl, e in slot_job.items():   # Reattach the watcher to an active run.
    if e in {j[0] for j in JOBS} and not done(e) and not failed(e):
        procs.append(subprocess.Popen([str(S / "auto_exp.sh"), e, sl, "noscore"]))
log("bắt đầu lại (chỉ CPU): started=%s packed=%s" % (sorted(started), sorted(packed)))
while True:
    for exp, _, _ in JOBS:
        if exp.startswith("cv_p") and exp not in packed and done(exp):
            if pack(exp):
                packed.add(exp)
    pending = [j for j in JOBS if j[0] not in started]
    if not pending and all(done(e) or failed(e) for e, _, _ in JOBS):
        log("xong 7 lượt trên Kaggle")
        break
    for slot in GPU + CPU:
        if not free(slot):
            continue
        is_gpu = slot in GPU
        ready = [j for j in pending if j[0] not in started and (j[2] is None or j[2] in packed)
                 and (is_gpu or not j[1])]
        if is_gpu:   # GPU priority: GPU-required runs first, then cv_f.
            ready.sort(key=lambda j: (not j[1], not j[0].startswith("cv_f")))
        if ready and launch(ready[0][0], slot, not is_gpu):
            time.sleep(20)
    time.sleep(120)
