#!/usr/bin/env python3
"""Install Sophia skills to user-level Grok, Cursor, and Claude Code directories.

Usage:
  python tools/install_skills.py --portable
  python tools/install_skills.py --all
  python tools/install_skills.py --all --cursor --claude
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORTABLE_SRC = ROOT / "skills" / "portable" / "sophia-source-discipline"
PROJECT_SKILL = ROOT / ".grok" / "skills" / "sophia-agi"


def copy_skill(src: Path, dest: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f"Installed {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Sophia AGI skills")
    parser.add_argument("--portable", action="store_true", help="Install portable source-discipline skill")
    parser.add_argument("--project", action="store_true", help="Copy project skill to ~/.grok/skills/sophia-agi")
    parser.add_argument("--all", action="store_true", help="Install portable + project skills")
    parser.add_argument("--cursor", action="store_true", help="Also copy portable skill to ~/.cursor/skills/")
    parser.add_argument("--claude", action="store_true", help="Also copy portable skill to ~/.claude/skills/")
    args = parser.parse_args()

    if not any((args.portable, args.project, args.all)):
        args.all = True

    home = Path.home()
    grok_skills = home / ".grok" / "skills"
    cursor_skills = home / ".cursor" / "skills"
    claude_skills = home / ".claude" / "skills"
    grok_skills.mkdir(parents=True, exist_ok=True)

    if args.all or args.portable:
        copy_skill(PORTABLE_SRC, grok_skills / "sophia-source-discipline")
        if args.cursor:
            cursor_skills.mkdir(parents=True, exist_ok=True)
            copy_skill(PORTABLE_SRC, cursor_skills / "sophia-source-discipline")
        if args.claude:
            claude_skills.mkdir(parents=True, exist_ok=True)
            copy_skill(PORTABLE_SRC, claude_skills / "sophia-source-discipline")

    if args.all or args.project:
        if PROJECT_SKILL.exists():
            copy_skill(PROJECT_SKILL, grok_skills / "sophia-agi")
        else:
            print(f"Skip project skill — missing {PROJECT_SKILL}")

    print("Reload Grok/Cursor skills (restart session or MCP panel).")
    print("Portable: /sophia-source-discipline or /source-discipline")
    print("Project:  /sophia-agi (when sophia-agi repo is workspace)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())