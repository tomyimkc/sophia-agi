---
description: View TrainWatch training stats — time/ETA/progress/loss (works locally + remote)
argument-hint: "[run-name-or-id]  (blank = all runs)"
allowed-tools: Bash(python3:*)
---
Current TrainWatch state:

!`python3 "$HOME/.claude/trainwatch_stats.py" $ARGUMENTS`

Relay the stats above concisely. Lead with **time/ETA and progress**, then loss/val_loss
trends; flag anything **stalled/anomalous**. On a remote machine with no local TrainWatch DB,
the script auto-targets the Spark dashboard (spark-2f2d:8420) over Tailscale; override with
$TRAINWATCH_URL. Don't re-run unless asked.
