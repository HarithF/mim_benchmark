#!/usr/bin/env bash
# Stage 3: run every built model's three binaries, check them against the
# PyTorch reference, and collect the timings.
#
#   pipeline/report.sh
#
# Output: $REPORTS_DIR/<run_id>/ with one <model>_<run_id>.json per model
# (reporter.py), timing.csv (timing_to_csv.py) and reporter.log.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

load_models
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="$REPORTS_DIR/$RUN_ID"
mkdir -p "$OUT"

log "report run $RUN_ID: $REPS rep(s), interleaved, pinned to CPU $PIN_CPU -> $OUT"
governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "?")
[[ "$governor" == performance ]] || echo "  note: CPU governor is '$governor' (sudo cpupower frequency-set -g performance)"

for k in "${MODELS[@]}"; do
  d="$MODELS_DIR/$(model_dir_name "$k")"
  if [[ ! -d "$d/bin" ]]; then
    echo "  skip $k: not built"
    continue
  fi
  # Each variant's driver forks one child per call and times only the call;
  # --interleave runs one rep of every variant per round.
  "$PYTHON" "$SCRIPTS/reporter.py" "$d" --reps "$REPS" --interleave --pin-cpu "$PIN_CPU" \
    --timeout "$RUN_TIMEOUT" --run-id "$RUN_ID" --report-dir "$OUT" >> "$OUT/reporter.log" 2>&1 || true
  "$PYTHON" - "$OUT/$(basename "$d")_$RUN_ID.json" "$k" <<'EOF'
import json, os, sys
path, k = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    print(f"  {k}: no report (see reporter.log)"); sys.exit()
r = json.load(open(path))
parts = []
for p, v in sorted(r["pipelines"].items()):
    t = (r.get("timing_summary") or {}).get(p, {}).get("median_ns")
    parts.append(f"{p} {v.get('status', '?')}" + (f" {t / 1e6:.1f} ms" if t else ""))
print(f"  {k}: ref={r.get('reference_source')}  " + ", ".join(parts))
EOF
done

"$PYTHON" "$SCRIPTS/timing_to_csv.py" "$OUT"/*_"$RUN_ID".json -o "$OUT/timing.csv" > /dev/null 2>&1 \
  && echo "  timings: $OUT/timing.csv" || echo "  timing_to_csv.py failed (reports are in $OUT)"
