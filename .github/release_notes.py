"""Print the release notes for one version: its changelog section and how to install it.

Usage: python .github/release_notes.py v0.1.0 owner/repository
"""

import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
SECTION = "## ["


def section(version: str) -> str:
    """The changelog text under the heading for ``version``, without the heading."""
    kept: list[str] = []
    inside = False
    for line in CHANGELOG.read_text(encoding="utf-8").splitlines():
        if line.startswith(SECTION):
            inside = line.startswith(f"{SECTION}{version}]")
            continue
        if inside:
            kept.append(line)
    return "\n".join(kept).strip()


def main() -> None:
    tag, repository = sys.argv[1], sys.argv[2]
    notes = section(tag.removeprefix("v"))
    if not notes:
        sys.exit(f"CHANGELOG.md has no section for {tag}")
    links = f"https://github.com/{repository}/releases/expanded_assets/{tag}"
    print("## Install\n")
    print("Python 3.10 or newer on Linux, macOS or Windows. No Rust toolchain is needed.\n")
    print(f"```bash\npip install mnemosoma --find-links {links}\n```\n")
    print("## What is in this version\n")
    print(notes)


if __name__ == "__main__":
    main()
