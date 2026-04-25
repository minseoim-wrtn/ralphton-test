#!/usr/bin/env python3
"""Simple TODO CLI tool."""

import argparse
import json
import sys
from pathlib import Path

TODO_FILE = Path("todos.json")


def load_todos():
    if TODO_FILE.exists():
        try:
            return json.loads(TODO_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            print(f"Error: {TODO_FILE} is corrupted. Fix or delete it.", file=sys.stderr)
            sys.exit(1)
    return []


def save_todos(todos):
    TODO_FILE.write_text(json.dumps(todos, ensure_ascii=False, indent=2) + "\n")


def cmd_add(args):
    todos = load_todos()
    todo = {"id": max((t["id"] for t in todos), default=0) + 1, "task": args.task, "done": False}
    todos.append(todo)
    save_todos(todos)
    print(f"Added: [{todo['id']}] {todo['task']}")


def cmd_list(args):
    todos = load_todos()
    if not todos:
        print("No todos yet. Add one with: python todo.py add \"task\"")
        return
    for t in todos:
        status = "x" if t["done"] else " "
        print(f"[{status}] {t['id']}. {t['task']}")


def cmd_done(args):
    todos = load_todos()
    for t in todos:
        if t["id"] == args.id:
            t["done"] = True
            save_todos(todos)
            print(f"Done: [{t['id']}] {t['task']}")
            return
    print(f"Error: todo #{args.id} not found.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Simple TODO CLI")
    sub = parser.add_subparsers(dest="command")

    p_add = sub.add_parser("add", help="Add a new todo")
    p_add.add_argument("task", help="Task description")

    sub.add_parser("list", help="List all todos")

    p_done = sub.add_parser("done", help="Mark a todo as done")
    p_done.add_argument("id", type=int, help="Todo ID to mark as done")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"add": cmd_add, "list": cmd_list, "done": cmd_done}[args.command](args)


if __name__ == "__main__":
    main()
