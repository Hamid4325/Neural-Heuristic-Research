"""Generate contribution table from git history and team role specifications.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import subprocess
import os


def get_git_contributors():
    try:
        out = subprocess.check_output(
            ["git", "shortlog", "-sne", "--all"],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception as e:
        return f"Error reading git shortlog: {e}"


def get_member_commits(author_query):
    try:
        out = subprocess.check_output(
            ["git", "log", f"--author={author_query}", "--pretty=format:%h %s", "-n", "10"],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip().split("\n") if out.strip() else []
    except Exception as e:
        return [f"Error: {e}"]


def main():
    print("=" * 60)
    print("               GIT SHORTLOG SUMMARY")
    print("=" * 60)
    print(get_git_contributors())
    print("\n" + "=" * 60)
    print("           RECENT COMMITS FOR ABDULLAH ZIA")
    print("=" * 60)
    for c in get_member_commits("Abdullah Zia"):
        print(f"  * {c}")
    print("=" * 60)


if __name__ == "__main__":
    main()
