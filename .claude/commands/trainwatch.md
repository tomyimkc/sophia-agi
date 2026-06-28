---
description: View TrainWatch training stats — time/ETA/progress/loss at a glance
argument-hint: "[run-name-or-id]  (blank = all runs)"
allowed-tools: Bash(python3:*)
---
Current TrainWatch state (DB: ~/.trainwatch/runs.db):

!`python3 /home/tomyimkc/sophia-agi/tools/trainwatch_stats.py $ARGUMENTS`

Relay the stats above to the user concisely. Lead with **time/ETA and progress**, then
loss/val_loss trends, and call out anything **stalled or anomalous**. If a specific run was
named, focus on its detail view. Don't re-run unless asked.
