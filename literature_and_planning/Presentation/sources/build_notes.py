"""Write speaker_notes.md from deck_content.py, the same source the deck's
embedded notes come from, and report the rehearsal estimate."""

from __future__ import annotations

import re
from pathlib import Path

import deck_content as C

RATE = 125.0   # words per minute, unhurried conference delivery with pauses

HEADER = """# Speaker notes — UAV remaining useful life estimation

Deck: `UAV_RUL_Project_20min.pptx` · Evidence cut-off: 8 September 2026

The narration below is embedded in the .pptx notes pane, slide for slide. The
"speaker reference" block under each slide is **not spoken**: it carries the
planned duration, the claim IDs from `evidence_ledger.csv`, and the artifact
the numbers came from.

Delivery notes:

- Do not read every number off a chart. The spoken text names the numbers that
  carry the argument; the rest are there for questions.
- Slides 4, 5, 7 and 10 carry the scope caveats. If time runs short, cut
  narration inside slides 3 and 6 rather than dropping either of those.
- Pause after slide 10's second panel; that is the point most audiences want a
  moment on.
- Backup slides B1-B6 follow the divider and are not part of the 20 minutes.

"""


def words(text):
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*", text))


def main():
    out = Path(__file__).resolve().parent.parent / "speaker_notes.md"
    lines = [HEADER]
    total_words = 0
    total_minutes = 0.0
    table = []
    for item in C.MAIN:
        n = item["n"]
        w = words(item["narration"])
        total_words += w
        total_minutes += item["minutes"]
        table.append((n, item["title"], item["minutes"], w, w / RATE))
        lines.append(f"## Slide {n} — {item['title']}\n")
        lines.append(f"*{item['subtitle']}*\n" if item.get("subtitle") else "")
        lines.append(item["narration"].strip() + "\n")
        lines.append("> **Speaker reference (not spoken).** "
                     f"Planned {item['minutes']:.2f} min · "
                     f"{w} words · about {w / RATE:.2f} min at {RATE:.0f} wpm.  \n"
                     f"> {item['source']}\n")
    lines.append("---\n")
    lines.append("## Backup slides (not timed)\n")
    for b in C.BACKUP:
        lines.append(f"**{b['n']} — {b['title']}**  \n{b['subtitle']}  \n"
                     f"{b['source']}\n")

    lines.append("---\n")
    lines.append("## Timing check\n")
    lines.append("| Slide | Topic | Planned min | Words | Spoken min at "
                 f"{RATE:.0f} wpm |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    for n, title, mins, w, est in table:
        lines.append(f"| {n} | {title} | {mins:.2f} | {w} | {est:.2f} |")
    lines.append(f"| | **Total** | **{total_minutes:.2f}** | **{total_words}** "
                 f"| **{total_words / RATE:.2f}** |")
    lines.append("")
    lines.append(f"Transitions and pauses: {20 - total_minutes:.2f} min. "
                 f"Planned content {total_minutes:.2f} min plus transitions = "
                 "20.00 min.")
    lines.append("")
    lines.append("The spoken-minute column is an estimate at "
                 f"{RATE:.0f} words per minute; it is not a substitute for a "
                 "rehearsal. Slides whose spoken estimate exceeds the planned "
                 "duration need either faster delivery or a sentence cut.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print(f"main-deck narration: {total_words} words, "
          f"{total_words / RATE:.2f} min at {RATE:.0f} wpm; "
          f"planned {total_minutes:.2f} min of content")
    for n, title, mins, w, est in table:
        flag = "  <-- over" if est > mins + 0.12 else ""
        print(f"  slide {n:>2}: planned {mins:.2f}  words {w:>3}  est {est:.2f}{flag}")


if __name__ == "__main__":
    main()
