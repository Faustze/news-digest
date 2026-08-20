"""
Daily commit: update LAST_BUILD.md / PROJECT_STATS.md via Groq LLM.

Runs only from 2026-08-20 to 2026-08-28 (inclusive, UTC). Gathers real
repository facts, asks the LLM to rewrite the state journal, validates the
answer, commits and pushes. Never fabricates progress: the model only
reformats facts collected from git/filesystem.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import groq
import httpx
import yaml
from langchain_groq import ChatGroq

REPO = Path(__file__).resolve().parent
START = dt.date(2026, 8, 20)
END = dt.date(2026, 8, 28)
STATE_FILES = ("LAST_BUILD.md", "PROJECT_STATS.md")


def _load_env() -> None:
    """Load KEY=VALUE pairs from .env if present (local runs only)."""
    env_file = REPO / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


PROMPT = """Ты обновляешь журнал состояния проекта в git-репозитории.

Тебе даны факты, собранные из репозитория прямо сейчас (git log, git status,
статистика файлов, текущее содержимое журналов). Твоя задача — переписать
файлы журнала на основе ЭТИХ ФАКТОВ.

Строгие правила:
- Не выдумывай работу, тесты, метрики, функциональность, которых нет в фактах.
- Если тесты/lint не запускались — честно напиши это.
- Формат LAST_BUILD.md: заголовок "# Last Build", поля Date/Branch/Commit,
  секции Current State, Recent Progress, Verification, Project Statistics,
  Next Focus (1-3 пункта).
- Формат PROJECT_STATS.md: сохрани существующую структуру файла.
- Commit message должен отражать реальное изменение, стиль:
  "docs: update project state" / "docs: update project statistics" /
  "docs: update project progress".
- Запрещены сообщения вида "chore: daily commit", "update: streak".

Ответь СТРОГО JSON-объектом без markdown-обёртки:
{
  "last_build": "<полное новое содержимое LAST_BUILD.md>",
  "stats": "<полное новое содержимое PROJECT_STATS.md или null, если файл не нужен/не меняется>",
  "commit_message": "<сообщение commit>"
}"""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def in_window(today: dt.date) -> bool:
    return START <= today <= END


def gather_facts() -> str:
    """Collect verifiable repository facts for the prompt."""
    py_files = [p for p in REPO.rglob("*.py") if ".venv" not in p.parts]
    loc = sum(p.read_text(errors="ignore").count("\n") + 1 for p in py_files)
    test_files = [
        p for p in py_files if p.name.startswith("test_") or "test" in p.parts
    ]
    tests = 0
    for p in test_files:
        tests += len(
            re.findall(
                r"^\s*(?:async\s+)?def\s+test_",
                p.read_text(errors="ignore"),
                re.MULTILINE,
            )
        )
    todos = 0
    for p in py_files:
        todos += len(re.findall(r"\b(?:TODO|FIXME)\b", p.read_text(errors="ignore")))

    existing = {}
    for name in STATE_FILES:
        path = REPO / name
        existing[name] = (
            path.read_text(errors="ignore")[:4000]
            if path.exists()
            else "(файл отсутствует)"
        )

    try:
        diff_stat = _git("diff", "--stat", "HEAD~1", "HEAD")
    except subprocess.CalledProcessError:
        diff_stat = "(недоступно)"

    return json.dumps(
        {
            "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head_commit": _git("log", "-1", "--oneline"),
            "recent_log": _git("log", "-15", "--date=short", "--pretty=%ad %h %s"),
            "status": _git("status", "--short") or "(чистый)",
            "last_commit_diff_stat": diff_stat,
            "stats": {
                "python_files": len(py_files),
                "python_loc": loc,
                "tests": tests,
                "todos_fixmes": todos,
            },
            "existing_last_build": existing["LAST_BUILD.md"],
            "existing_project_stats": existing["PROJECT_STATS.md"],
        },
        ensure_ascii=False,
    )


def ask_llm(facts: str) -> dict:
    """Ask Groq to regenerate the state journal. Returns parsed JSON."""
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    model = config.get("model") or "llama-3.3-70b-versatile"
    llm = ChatGroq(model=model, temperature=0, max_tokens=8192)

    raw = llm.invoke([("system", PROMPT), ("human", f"Факты репозитория:\n{facts}")])
    content = raw.content if hasattr(raw, "content") else str(raw)
    clean = re.sub(r"```(?:json)?|```", "", content).strip()
    parsed = json.loads(clean)
    if not isinstance(parsed, dict) or "last_build" not in parsed:
        raise ValueError("LLM answer missing required fields")
    return parsed


def validate(answer: dict) -> None:
    lb = answer.get("last_build") or ""
    if not lb.strip() or "# Last Build" not in lb:
        raise ValueError("generated LAST_BUILD.md is empty or malformed")
    msg = answer.get("commit_message") or ""
    if not msg.strip() or msg.strip() in {
        "chore: daily commit",
        "update: streak",
        "keep contribution active",
    }:
        raise ValueError("commit message missing or forbidden")


def apply_and_commit(answer: dict) -> str | None:
    """Write files, commit and push. Returns commit hash or None."""
    changed = False
    (REPO / "LAST_BUILD.md").write_text(answer["last_build"].strip() + "\n")
    changed = True

    stats = answer.get("stats")
    if isinstance(stats, str) and stats.strip():
        (REPO / "PROJECT_STATS.md").write_text(stats.strip() + "\n")
        changed = True

    if not changed:
        return None

    _git("add", *STATE_FILES)
    if (
        subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=REPO, check=False
        ).returncode
        == 0
    ):
        return None

    _git("commit", "-m", answer["commit_message"].strip())
    _git("push")
    return _git("rev-parse", "--short", "HEAD")


def main() -> int:
    _load_env()
    today = dt.datetime.now(dt.timezone.utc).date()
    if not in_window(today):
        print(f"[daily-commit] {today} outside {START}..{END}; nothing to do.")
        return 0

    facts = gather_facts()
    try:
        answer = ask_llm(facts)
    except json.JSONDecodeError as e:
        print(f"[daily-commit] FATAL: LLM returned invalid JSON: {e}")
        return 1
    except (groq.GroqError, httpx.HTTPError, TimeoutError) as e:
        print(f"[daily-commit] FATAL: Groq request failed: {e}")
        return 1

    validate(answer)
    commit = apply_and_commit(answer)
    if commit:
        print(f"[daily-commit] pushed {commit}: {answer['commit_message']}")
    else:
        print("[daily-commit] no meaningful change; nothing committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
