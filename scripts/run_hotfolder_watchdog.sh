#!/usr/bin/env bash
# Resilient launcher for the image-to-splat hot-folder daemon.
#
# WHAT IT SURVIVES:
#   - Daemon OOM kills (host RSS leak crosses WSL cap -> silent SIGKILL 137)
#   - GPU driver resets from pathological assets ("device not ready", exit 2)
#   - The watchdog process itself dying (PC reboot, window closed)
#
# HOW:
#   - Relaunches the daemon and resumes via skip-existing (finished datasets
#     are never redone).
#   - PER-IMAGE 3-STRIKE QUARANTINE: each death strikes the in-flight image,
#     identified from THIS launch's new log lines (FAIL line first, then the
#     batch/probe/single in-flight marker — works in every mode). Under 3
#     strikes -> retried; at 3 -> moved to failed/ (found recursively, incl.
#     T#_seed#/ batch folders) and bypassed; the run continues on the rest.
#   - PERSISTENT STATE: the chosen run config + strike counts are written to
#     $HF/.watchdog_state after every change. On startup the watchdog READS
#     it, so even after a reboot it resumes the SAME run (no re-prompt) with
#     strike history intact. State is cleared when the run finishes.
#
# USAGE:
#   run_hotfolder_watchdog.sh --ask                    # prompt once (or resume), then run
#   run_hotfolder_watchdog.sh --tier 6 --probe-seed-count 3 --seed 222
#   run_hotfolder_watchdog.sh --batch-tiered '/abs/path/to/inbox'
#
# Ctrl+C once to stop. State is kept if work remains (so you can resume),
# cleared if the queue is drained.

cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENCV_IO_ENABLE_OPENEXR=1

# Optional leading "--hotfolder PATH": the self-locating BAT passes the folder
# it lives in, so the daemon, dashboard, and watchdog all agree on one root.
# Falls back to $IMAGE2SPLAT_HOTFOLDER, then ~/image2splat.
HF="${IMAGE2SPLAT_HOTFOLDER:-$HOME/image2splat}"
if [ "${1:-}" = "--hotfolder" ]; then HF="$2"; shift 2; fi
HF_ARG=(--hotfolder "$HF")
PY="$HOME/miniconda3/envs/trellis2/bin/python"
LOG="$HF/daemon.log"
STATE="$HF/.watchdog_state"     # TAB-delimited: "RUNARGS\t<args...>" + "STRIKE\t<name>\t<n>"
DELAY=8
MAX_STRIKES=3

declare -A strikes
RUNARGS=()

state_save() {
    { printf 'RUNARGS'; printf '\t%s' "${RUNARGS[@]}"; printf '\n'
      local k
      for k in "${!strikes[@]}"; do printf 'STRIKE\t%s\t%s\n' "$k" "${strikes[$k]}"; done
    } > "$STATE" 2>/dev/null
}
state_load() {                   # populates RUNARGS + strikes from $STATE
    RUNARGS=(); strikes=()
    local f
    while IFS=$'\t' read -r -a f; do
        case "${f[0]}" in
            RUNARGS) RUNARGS=("${f[@]:1}") ;;
            STRIKE)  [ -n "${f[1]:-}" ] && strikes["${f[1]}"]="${f[2]:-1}" ;;
        esac
    done < "$STATE"
}
prompt_config() {                # interactive run picker -> RUNARGS (all flows)
    echo "============================================================"
    echo "  Run type:"
    echo "    1) Default   2) Subtle   3) Balanced   4) Refined   5) Sculpted"
    echo "    6) Probe   (all 5 tiers at view 129)"
    echo "    B) Batch   (run pre-staged T#_seed#/ folders in inbox)"
    local _t _s _sc
    while true; do
        read -rp "  Select [1-6 / B, default 1]: " _t; _t="${_t:-1}"
        case "$_t" in 1|2|3|4|5|6|b|B) break;; *) echo "  invalid, pick 1-6 or B";; esac
    done
    if [[ "$_t" =~ ^[bB]$ ]]; then
        RUNARGS=(--batch-tiered "$HF/inbox")     # tier+seed come from folder names
        return
    fi
    read -rp "  Anchor seed [default 222]: " _s; _s="${_s:-222}"
    RUNARGS=(--tier "$_t" --seed "$_s")
    if [ "$_t" = "6" ]; then
        echo "  Probe seeds: enter a COUNT for random (e.g. 3), or an explicit"
        echo "  comma-list to PIN exact seeds (e.g. 222,74964,91766)."
        local _sc
        while true; do
            read -rp "  Seeds [default 3]: " _sc; _sc="${_sc:-3}"
            if [[ "$_sc" =~ ^[0-9]+(,[0-9]+)+$ ]]; then
                RUNARGS+=(--probe-seeds "$_sc"); break          # explicit pinned list
            elif [[ "$_sc" =~ ^[1-8]$ ]]; then
                RUNARGS+=(--probe-seed-count "$_sc"); break      # random count
            else
                echo "  invalid — a count 1-8, or a comma-list like 222,74964,91766"
            fi
        done
    fi
}

# --- Resolve run config: resume saved state, prompt, or take explicit args ---
if [ "${1:-}" = "--ask" ]; then
    if [ -f "$STATE" ]; then
        state_load
        echo "Found an interrupted run: ${RUNARGS[*]}  (${#strikes[@]} image(s) with strikes)"
        read -rp "Resume it? [Y/n]: " _yn
        if [[ "$_yn" =~ ^[Nn] ]]; then
            strikes=(); prompt_config
        else
            echo "Resuming previous run."
        fi
    else
        prompt_config
    fi
else
    RUNARGS=("$@")               # explicit args win; inherit strike history if any
    if [ -f "$STATE" ]; then
        _ra=("${RUNARGS[@]}"); state_load; RUNARGS=("${_ra[@]}")
    fi
fi
state_save
echo "[watchdog] config: --no-prompt ${RUNARGS[*]}   (state: $STATE)"

stop=0
trap 'stop=1' INT TERM

# One-time: rescue any image stranded in processing/ by a previous crash.
shopt -s nullglob
for f in "$HF"/processing/*; do
    [ -f "$f" ] || continue
    case "$f" in *.png|*.jpg|*.jpeg|*.webp|*.bmp)
        mv -f "$f" "$HF/inbox/" 2>/dev/null \
            && echo "[watchdog] rescued pre-existing stray -> inbox: $(basename "$f")" ;;
    esac
done
shopt -u nullglob

attempt=0
fastfail=0
while [ "$stop" -eq 0 ]; do
    attempt=$((attempt + 1))
    echo "[watchdog] $(date '+%F %T')  launch #$attempt  (--no-prompt ${HF_ARG[*]} ${RUNARGS[*]})"
    start=$(date +%s)
    loglines_before=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    "$PY" -u scripts/hotfolder_daemon.py --no-prompt "${HF_ARG[@]}" "${RUNARGS[@]}"
    code=$?
    ran=$(( $(date +%s) - start ))
    echo "[watchdog] $(date '+%F %T')  daemon exited code=$code (ran ${ran}s)"

    # Clean, intentional exits -> done. Clear state, stop.
    if [ "$code" -eq 0 ] || [ "$code" -eq 1 ]; then
        echo "[watchdog] run finished — clearing state."
        rm -f "$STATE"; break
    fi
    # Ctrl+C / SIGTERM -> stop but KEEP state if inbox still has work.
    if [ "$stop" -eq 1 ] || [ "$code" -eq 130 ] || [ "$code" -eq 143 ]; then
        if ! ls "$HF"/inbox/* "$HF"/inbox/*/* 2>/dev/null | grep -qiE '\.(png|jpg|jpeg|webp|bmp)$'; then
            rm -f "$STATE"; echo "[watchdog] stopped, queue empty — state cleared."
        else
            echo "[watchdog] stopped — state kept; re-run to resume."
        fi
        break
    fi

    # --- per-image strike accounting (persisted) -------------------------
    new=$(tail -n "+$((loglines_before + 1))" "$LOG" 2>/dev/null)
    culprit=$(printf '%s\n' "$new" \
              | grep -aE "FAIL [^:]+\.(png|jpg|jpeg|webp|bmp):" | tail -1 \
              | sed -E 's/.*FAIL +(.+\.(png|jpg|jpeg|webp|bmp)):.*/\1/')
    if [ -z "$culprit" ]; then
        culprit=$(printf '%s\n' "$new" \
                  | grep -aE "\][[:space:]]+(START|INFER|PROBE)[[:space:]]|\[[0-9]+\.[0-9]+/[0-9]+\][[:space:]]" | tail -1 \
                  | sed -E 's/.*\[[0-9]+\.[0-9]+\/[0-9]+\][[:space:]]+//;
                            s/.*\][[:space:]]*(START|INFER|PROBE)[[:space:]]+//;
                            s/[[:space:]]*→[[:space:]]*slug=.*//')
    fi
    if [ -n "$culprit" ]; then
        strikes["$culprit"]=$(( ${strikes["$culprit"]:-0} + 1 ))
        n=${strikes["$culprit"]}
        cpath=$(find "$HF/inbox" "$HF/processing" -maxdepth 2 -type f -name "$culprit" 2>/dev/null | head -1)
        if [ "$n" -ge "$MAX_STRIKES" ]; then
            [ -n "$cpath" ] && mv -f "$cpath" "$HF/failed/" 2>/dev/null
            [ -f "$HF/processing/$culprit" ] && mv -f "$HF/processing/$culprit" "$HF/failed/" 2>/dev/null
            echo "[watchdog] '$culprit' failed ${n}x — QUARANTINED to failed/, bypassing."
        else
            [ -f "$HF/failed/$culprit" ] && mv -f "$HF/failed/$culprit" "$HF/inbox/" 2>/dev/null
            echo "[watchdog] '$culprit' strike ${n}/${MAX_STRIKES} — will retry."
        fi
        state_save
    fi

    if [ "$ran" -lt 180 ]; then fastfail=$((fastfail + 1)); else fastfail=0; fi
    if [ "$fastfail" -ge 5 ]; then
        echo "[watchdog] 5 instant crashes in a row — systemic failure, stopping (state kept). Check daemon.log."
        break
    fi

    echo "[watchdog] relaunching in ${DELAY}s. Ctrl+C to stop."
    sleep "$DELAY" || break
done

echo "[watchdog] stopped after $attempt launch(es)."
