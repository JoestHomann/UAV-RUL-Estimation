"""Build UAV_RUL_Project_20min.pptx on a copy of the supplied University of
Stuttgart template.

The template's masters, layouts, theme colours, fonts and logo are reused
unchanged; only the ten sample slides are removed. Every scientific chart is a
figure the repository already generated (see ``figure_manifest.csv``); the only
native PowerPoint drawings are the nested-split schematic, the retained-model
schematic and the tables, because the repository has no figure for those.

    python build_deck.py --template <path to the supplied .pptx> --out <path>
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

import deck_content as C

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

# --- template palette ("UNI COLOUR") ---------------------------------------
DARK = RGBColor(0x3E, 0x44, 0x4C)
BLUE = RGBColor(0x00, 0x51, 0x9E)
CYAN = RGBColor(0x00, 0xBE, 0xFF)
LBLUE = RGBColor(0x7D, 0xC6, 0xEA)
GREY = RGBColor(0x9F, 0x99, 0x98)
YELLOW = RGBColor(0xFF, 0xD5, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xEC, 0xF4, 0xFA)

# --- geometry (inches) on the template's 10 x 5.625 in canvas ---------------
L, R = 0.51, 9.53
CONTENT_TOP, CONTENT_BOTTOM = 1.06, 5.16
W = R - L
FOOTER_TEXT = "UAV remaining useful life estimation"
FOOTER_DATE = "8 September 2026"

LAYOUT = {}


# ------------------------------------------------------------------ utils --
def in_(value):
    return Inches(value)


def set_text(frame, blocks, size=14, color=DARK, bold=False, space_after=4,
             align=PP_ALIGN.LEFT, line=None):
    frame.word_wrap = True
    first = True
    for block in blocks:
        text, level = (block, 0) if isinstance(block, str) else block
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.level = level
        paragraph.alignment = align
        paragraph.space_after = Pt(space_after)
        if line:
            paragraph.line_spacing = line
        run = paragraph.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "Arial"


def textbox(slide, x, y, w, h, blocks, **kwargs):
    box = slide.shapes.add_textbox(in_(x), in_(y), in_(w), in_(h))
    box.text_frame.word_wrap = True
    set_text(box.text_frame, blocks, **kwargs)
    return box


def rounded(slide, x, y, w, h, fill, line=None,
            shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    element = slide.shapes.add_shape(shape, in_(x), in_(y), in_(w), in_(h))
    element.fill.solid()
    element.fill.fore_color.rgb = fill
    if line is None:
        element.line.fill.background()
    else:
        element.line.color.rgb = line
        element.line.width = Pt(1)
    element.shadow.inherit = False
    element.text_frame.word_wrap = True
    element.text_frame.margin_left = Emu(45720)
    element.text_frame.margin_right = Emu(45720)
    element.text_frame.margin_top = Emu(27432)
    element.text_frame.margin_bottom = Emu(27432)
    return element


def headed_box(slide, x, y, w, h, head, lines, fill=WHITE, line=GREY,
               size=10, head_color=BLUE):
    box = rounded(slide, x, y, w, h, fill, line=line)
    set_text(box.text_frame, [(head, 0)] + [(t, 0) for t in lines],
             size=size, color=DARK, space_after=2)
    box.text_frame.paragraphs[0].runs[0].font.bold = True
    box.text_frame.paragraphs[0].runs[0].font.color.rgb = head_color
    return box


def place_picture(slide, name, box, top_align=False):
    path = FIG / name
    width_px, height_px = Image.open(path).size
    x, y, w, h = box
    scale = min(w / width_px, h / height_px)
    pw, ph = width_px * scale, height_px * scale
    px = x + (w - pw) / 2
    py = y if top_align else y + (h - ph) / 2
    return slide.shapes.add_picture(str(path), in_(px), in_(py), in_(pw),
                                    in_(ph))


def copy_hf_placeholders(slide, layout):
    tree = slide.shapes._spTree
    for shape in layout.placeholders:
        kind = str(shape.placeholder_format.type)
        if not kind.startswith(("DATE", "FOOTER", "SLIDE_NUMBER")):
            continue
        tree.append(copy.deepcopy(shape._element))
    for shape in slide.shapes:
        if not shape.is_placeholder:
            continue
        kind = str(shape.placeholder_format.type)
        if kind.startswith("FOOTER"):
            shape.text_frame.paragraphs[0].runs[0].text = FOOTER_TEXT
        elif kind.startswith("DATE"):
            shape.left, shape.top = Inches(7.10), Inches(5.33)
            shape.width, shape.height = Inches(1.90), Inches(0.16)
            for paragraph in shape.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.RIGHT
                for run in paragraph.runs:
                    run.text = FOOTER_DATE


def new_slide(prs, layout_name, title, subtitle, notes, with_hf=True,
              title_kw=None, keep_idx=()):
    layout = LAYOUT[layout_name]
    slide = prs.slides.add_slide(layout)
    title_kw = title_kw or dict(size=18, bold=True, color=DARK)
    for placeholder in list(slide.placeholders):
        index = placeholder.placeholder_format.idx
        kind = str(placeholder.placeholder_format.type)
        if kind.startswith("TITLE") or kind.startswith("CENTER_TITLE"):
            placeholder.text_frame.word_wrap = True
            set_text(placeholder.text_frame, [title], **title_kw)
        elif index == 13 and subtitle is not None:
            set_text(placeholder.text_frame, [subtitle], size=14, color=GREY)
        elif index in keep_idx:
            continue
        else:
            element = placeholder._element
            element.getparent().remove(element)
    if with_hf:
        copy_hf_placeholders(slide, layout)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def source_label(slide, text, y=None):
    y = CONTENT_BOTTOM - 0.20 if y is None else y
    textbox(slide, L, y, W, 0.18, [text], size=9, color=GREY)


def notes_text(item):
    return (item["narration"].strip() + "\n\n"
            + "----- speaker reference, not spoken -----\n"
            + f"Planned duration: {item['minutes']:.2f} min\n"
            + item.get("source", ""))


def table(slide, x, y, w, headers, rows, widths, size=9.5, row_height=0.24,
          highlight=(), header_fill=BLUE):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), in_(x), in_(y),
                                   in_(w), in_(row_height * (len(rows) + 1)))
    grid = shape.table
    for index, width in enumerate(widths):
        grid.columns[index].width = in_(width)
    for index, header in enumerate(headers):
        cell = grid.cell(0, index)
        cell.text = ""
        set_text(cell.text_frame, [header], size=size, bold=True, color=WHITE,
                 space_after=0)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for row_index, row in enumerate(rows, start=1):
        for column_index, value in enumerate(row):
            cell = grid.cell(row_index, column_index)
            cell.text = ""
            marked = (row_index - 1) in highlight
            set_text(cell.text_frame, [str(value)], size=size,
                     color=BLUE if marked else DARK, bold=marked,
                     space_after=0)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PALE if row_index % 2 else WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for row in grid.rows:
        row.height = in_(row_height)
    return grid


# ------------------------------------------------------- slide builders ----
def slide_title(prs, item):
    """Template title-slide design: dark title panel right, presenter circle,
    layout logo. The layout's photograph placeholder is left empty and the
    prediction-timeline figure occupies the free left half."""
    slide = new_slide(prs, "Titelfolie", item["title"], None, notes_text(item),
                      with_hf=False,
                      title_kw=dict(size=24, bold=True, color=WHITE),
                      keep_idx=(10, 11))
    for placeholder in list(slide.placeholders):
        if placeholder.placeholder_format.idx == 11:
            set_text(placeholder.text_frame, ["Joest Homann"], size=12,
                     bold=True, color=DARK, align=PP_ALIGN.CENTER,
                     space_after=0)
            placeholder.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    textbox(slide, 0.30, 1.60, 4.72, 0.54, [item["subtitle"]], size=13,
            color=BLUE, bold=True)
    place_picture(slide, item["figure"], (0.22, 2.18, 4.88, 2.42))
    textbox(slide, 0.30, 4.66, 4.72, 0.24,
            ["University of Stuttgart · 8 September 2026"], size=10,
            color=GREY)
    textbox(slide, 0.30, 4.92, 4.72, 0.20, [item["source"]], size=8,
            color=GREY)
    return slide


def slide_figure(prs, item):
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      notes_text(item))
    place_picture(slide, item["figure"], (L, CONTENT_TOP + 0.04, W, 3.78))
    source_label(slide, item["source"])
    return slide


def slide_split_diagram(prs, item):
    """Phase 1: nested whole-UAV design (no repository figure) beside the
    repository's train/test history-length figure."""
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      notes_text(item))
    left_width = 4.30

    textbox(slide, L, 1.08, left_width, 0.22,
            ["Nested whole-UAV design, all five outer folds"],
            size=11, bold=True, color=BLUE)

    row_y = 1.36
    fold_width = (left_width - 4 * 0.05) / 5
    for index in range(5):
        x = L + index * (fold_width + 0.05)
        held = index == 0
        box = rounded(slide, x, row_y, fold_width, 0.42,
                      YELLOW if held else BLUE, shape=MSO_SHAPE.RECTANGLE)
        set_text(box.text_frame, [f"fold {index + 1}\n20 UAVs"], size=8,
                 color=DARK if held else WHITE, align=PP_ALIGN.CENTER,
                 space_after=0)
        box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    textbox(slide, L, 1.82, left_width, 0.20,
            ["fold 1 held out; folds 2-5 are the 80 training UAVs"],
            size=9, color=DARK)

    textbox(slide, L, 2.12, left_width, 0.22,
            ["Inside those 80: four inner folds select everything"],
            size=11, bold=True, color=BLUE)
    inner_y = 2.40
    inner_width = (left_width - 3 * 0.05) / 4
    for index in range(4):
        x = L + index * (inner_width + 0.05)
        box = rounded(slide, x, inner_y, inner_width, 0.38,
                      CYAN if index == 0 else LBLUE, shape=MSO_SHAPE.RECTANGLE)
        set_text(box.text_frame, [f"inner {index + 1}"], size=8, color=DARK,
                 align=PP_ALIGN.CENTER, space_after=0)
        box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    textbox(slide, L, 2.82, left_width, 0.20,
            ["features, hyperparameters, early stopping, blend weights"],
            size=9, color=DARK)

    headed_box(slide, L, 3.12, left_width, 1.62, "Rules that follow", [
        "· Every prefix of one UAV stays in one fold.",
        "· Features use only cycles at or before the cutoff.",
        "· Scalers refitted inside each training partition.",
        "· Each UAV carries total training weight 1.",
        "· Uncertainty resamples whole UAVs, never rows.",
    ], size=10)

    place_picture(slide, item["figure"],
                  (L + left_width + 0.22, 1.30, W - left_width - 0.22, 3.10))
    source_label(slide, item["source"])
    return slide


def slide_ensemble(prs, item):
    """Retained-model schematic (no repository figure) beside PE_11."""
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      notes_text(item))
    top = CONTENT_TOP + 0.06
    diagram_width = 3.62

    def box(x, y, w, h, text, fill, color=WHITE, size=9.5, bold=False):
        element = rounded(slide, x, y, w, h, fill)
        set_text(element.text_frame, text.split("\n"), size=size, color=color,
                 align=PP_ALIGN.CENTER, space_after=0, bold=bold)
        element.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        return element

    half = (diagram_width - 0.10) / 2
    box(L, top, half, 0.40, "3 × XGBoost\nseeds 13/37/73", BLUE, size=8.5)
    box(L + half + 0.10, top, half, 0.40, "3 × ExtraTrees\nseeds 13/37/73",
        BLUE, size=8.5)
    box(L, top + 0.46, half, 0.28, "average", LBLUE, color=DARK, size=8.5)
    box(L + half + 0.10, top + 0.46, half, 0.28, "average", LBLUE, color=DARK,
        size=8.5)
    box(L, top + 0.80, diagram_width, 0.36,
        "family blend weight selected on\ntraining-side data only", CYAN,
        color=DARK, size=8.5)
    box(L, top + 1.22, diagram_width, 0.28, "base prediction", GREY, size=9,
        bold=True)
    box(L, top + 1.56, diagram_width, 0.80,
        "residual model  (7 leaves, 100 iterations)\ninputs: base prediction · "
        "observed history length · member spread and range · family "
        "disagreement · 8 sensor baseline-deltas and slopes",
        YELLOW, color=DARK, size=8)
    box(L, top + 2.42, diagram_width, 0.32,
        "final RUL = max(0, base − residual)", BLUE, size=9, bold=True)

    warning = rounded(slide, L, top + 2.80, diagram_width, 0.74, WHITE,
                      line=GREY)
    set_text(warning.text_frame,
             ["Out-of-fold calibration predictions come from models that "
              "excluded that UAV; calibration endpoints are filtered to "
              "training UAVs. The true future RUL is never an input."],
             size=8, color=DARK, space_after=0)
    warning.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    place_picture(slide, item["figure"],
                  (L + diagram_width + 0.24, top + 0.10,
                   W - diagram_width - 0.24, 3.00))
    textbox(slide, L + diagram_width + 0.24, top + 3.18,
            W - diagram_width - 0.24, 0.44,
            ["11.5080 → 11.0243 cycles, 4.20% better, 5 of 5 fold wins. "
             "Refitted by the Phase 3 Run 7 adapter on all 100 training UAVs."],
            size=8.5, color=DARK)
    source_label(slide, item["source"])
    return slide


AFTER_RUN_7 = [
    ["PE_20  nested refit", "+1.56% worse", "2/5", "rejected"],
    ["PE_18  TabPFN blend", "−1.30%", "4/5", "to confirm"],
    ["PE_21  two new seeds", "−1.12%", "6/10", "rejected"],
    ["PE_24  regime gate", "−1.90%", "11/15", "rejected"],
    ["PE_26  short history", "−0.39%", "6/15", "rejected"],
    ["PE_27  UAV subsets", "screen only", "3/5", "stopped"],
]


def slide_screen(prs, item):
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      notes_text(item))
    figure_width = 5.15
    place_picture(slide, item["figure"],
                  (L, CONTENT_TOP + 0.10, figure_width, 3.40))
    textbox(slide, L, CONTENT_TOP + 3.52, figure_width, 0.24,
            ["PE_15 screen: 11.02 → 9.59 cycles, −13.0%, 5 of 5 folds"],
            size=9.5, color=BLUE, bold=True)

    x = L + figure_width + 0.24
    width = R - x
    textbox(slide, x, CONTENT_TOP + 0.04, width, 0.22,
            ["After Run 7: six candidates, all gated"],
            size=10.5, bold=True, color=BLUE)
    table(slide, x, CONTENT_TOP + 0.32, width,
          ["candidate", "mean RMSE", "wins", "outcome"], AFTER_RUN_7,
          [1.42, 0.83, 0.50, 0.83], size=8, row_height=0.26)
    box = rounded(slide, x, CONTENT_TOP + 2.34, width, 1.00, YELLOW)
    set_text(box.text_frame,
             ["Every promotion gate is declared before the run. Nothing passed, "
              "so nothing was submitted, so the public score is still 0.87652."],
             size=9.5, color=DARK, space_after=0)
    box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    source_label(slide, item["source"])
    return slide


PE28_ROWS = [
    ("A_current", "298", "current pipeline representation (control)"),
    ("B_other", "266", "alternative script: 22 sensors, simpler summaries"),
    ("C_other_14", "170", "alternative summaries on 14 sensors"),
    ("D_long_windows", "266", "longer trailing windows"),
    ("E_compact_22", "134", "compact summaries, 22 sensors"),
    ("F_baseline_10", "266", "10-cycle baseline averaging"),
    ("G_all_windows", "310", "all window lengths retained"),
    ("H_compact_14", "86", "compact summaries, 14 sensors"),
    ("I_four_sensors", "50", "four sensors only"),
    ("J_recent_slopes", "310", "added recent-slope features"),
]


def slide_table(prs, item):
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      notes_text(item))
    top = CONTENT_TOP + 0.12
    width = 5.75
    table(slide, L, top, width, ["feature set", "features", "what it varies"],
          [list(row) for row in PE28_ROWS], [1.55, 0.78, width - 2.33],
          size=9, row_height=0.235, highlight={0})

    x = L + width + 0.26
    right = R - x
    headed_box(slide, x, top, right, 1.00, "Crossed with two model recipes", [
        "· the retained Run 7 ensemble",
        "· an XGBoost / CatBoost blend",
        "100 screening evaluations, seed 20270107",
    ])
    headed_box(slide, x, top + 1.12, right, 1.02, "Gates, declared in advance", [
        "· screen: 2% mean-RMSE gain, 4/5 folds",
        "· confirm: 2% gain, 8/10 folds, pooled R² ≥ 0.90",
        "  and a bootstrap interval wholly below zero",
    ])
    status = rounded(slide, x, top + 2.26, right, 1.28, YELLOW)
    set_text(status.text_frame,
             [("Status on 8 September 2026", 0),
              ("PENDING — training started, no complete winner manifest.", 0),
              ("Motivated by a user-reported public score of 0.87877 for a "
               "separate script (266 features, 22 sensors). That is not Run 7's "
               "score and not a matched experiment.", 0)],
             size=9.5, color=DARK, space_after=2)
    status.text_frame.paragraphs[0].runs[0].font.bold = True
    source_label(slide, item["source"])
    return slide


CHAIN = [
    ["Phase 1 + 2", "Bounded validation scenarios with a fitting cap at 125",
     "Run 3 → 4", "+0.30377"],
    ["Phase 0 + 2", "Drift-pruned 298 features, calibrated 50/50 tree blend",
     "Run 4 → 5", "+0.02012"],
    ["Phase 2", "Conditional conservative calibration at q = 0.55",
     "Run 5 → 6", "+0.00216"],
    ["Phase 2", "Cross-fitted residual correction on six seeded members",
     "Run 6 → 7", "+0.00911"],
]


def slide_conclusions(prs, item):
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      notes_text(item))
    top = CONTENT_TOP + 0.12
    table(slide, L, top, W,
          ["phase", "decision", "submissions", "public R² change"],
          CHAIN, [1.30, 4.90, 1.30, 1.52], size=10.5, row_height=0.36,
          highlight={0})

    note = rounded(slide, L, top + 1.98, W, 0.62, WHITE, line=GREY)
    set_text(note.text_frame,
             ["Each row is a submission-to-submission difference, not an "
              "isolated ablation: a run can carry more than one change. "
              "Evaluation and target design dominated model choice."],
             size=10.5, color=DARK, space_after=0)
    note.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    open_box = rounded(slide, L, top + 2.70, W, 0.96, YELLOW)
    set_text(open_box.text_frame,
             [("What we do not claim", 0),
              ("Development mean-fold R² is 0.90041; the recorded public score "
               "is 0.87652. Closing that needs about 10.0% lower RMSE on the "
               "same scored set, and no candidate after Run 7 passed a nested "
               "confirmation.", 0)],
             size=10.5, color=DARK, space_after=2)
    open_box.text_frame.paragraphs[0].runs[0].font.bold = True
    open_box.text_frame.paragraphs[0].runs[0].font.size = Pt(11.5)
    source_label(slide, item["source"])
    return slide


# ------------------------------------------------------------- backups -----
def divider(prs):
    slide = prs.slides.add_slide(LAYOUT["Kapitel"])
    for placeholder in list(slide.placeholders):
        kind = str(placeholder.placeholder_format.type)
        if kind.startswith("TITLE") or kind.startswith("CENTER_TITLE"):
            set_text(placeholder.text_frame, ["Backup slides"], size=24,
                     bold=True, color=DARK)
        else:
            set_text(placeholder.text_frame,
                     ["Not part of the 20-minute talk. Held for questions."],
                     size=14, color=GREY)
    slide.notes_slide.notes_text_frame.text = (
        "End of the main deck. Six backup slides follow: the full locked "
        "architecture table, the sequence and hybrid studies, the complete "
        "cap/scenario matrix, residual-correction and calibration detail, "
        "rejected representation experiments, and uncertainty and score "
        "provenance.")
    return slide


def build_backups(prs):
    items = {item["n"]: item for item in C.BACKUP}

    # --- B1 full locked architecture comparison
    item = items["B1"]
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      "Use if asked whether the sequence models were given a "
                      "fair chance, or about seed stability.")
    table(slide, L, CONTENT_TOP + 0.10, W,
          ["architecture", "seeds", "R² mean", "RMSE mean", "R² seed SD",
           "R² 95% UAV-bootstrap interval"],
          [["XGBoost", "3", "0.8041", "28.4989", "0.0055", "0.7528 - 0.8437"],
           ["Random Forest", "3", "0.7827", "30.0138", "0.0030", "0.7058 - 0.8348"],
           ["Extra Trees", "3", "0.7640", "31.2779", "0.0027", "0.6671 - 0.8372"],
           ["Trajectory DTW-kNN", "1", "0.5067", "45.2239", "0", "0.3804 - 0.6106"],
           ["Sensor-graph TCN", "3", "0.4294", "48.2278", "0.1487", "0.1602 - 0.6359"],
           ["Multi-scale CNN", "3", "0.3607", "51.4209", "0.0622", "0.2317 - 0.4691"],
           ["Mean baseline", "1", "-0.3066", "73.6023", "0", "-0.4848 - -0.1758"],
           ["TCN", "3", "-2.4231", "111.2088", "2.1691", "-9.7147 - 0.6382"]],
          [2.10, 0.62, 0.92, 1.02, 1.00, 3.36], highlight={0})
    textbox(slide, L, CONTENT_TOP + 2.42, W, 0.90, [item["note"]], size=10,
            color=DARK)
    source_label(slide, item["source"])

    # --- B2 sequence and hybrid models
    item = items["B2"]
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      "Use if challenged that trees were preferred without a "
                      "fair rematch. The hybrid models receive the raw window "
                      "and all 298 engineered features.")
    figure_width = 5.70
    place_picture(slide, item["figure"],
                  (L, CONTENT_TOP + 0.12, figure_width, 3.10))
    textbox(slide, L, CONTENT_TOP + 3.28, figure_width, 0.44,
            ["Architecture study run 8, early/middle scenarios and cap-125 "
             "target. Neither hybrid won a single fold out of five."],
            size=9, color=DARK)
    x = L + figure_width + 0.24
    width = R - x
    textbox(slide, x, CONTENT_TOP + 0.06, width, 0.22,
            ["Dedicated temporal study (run 7)"], size=10.5, bold=True,
            color=BLUE)
    table(slide, x, CONTENT_TOP + 0.34, width, ["model", "mean R²", "mean RMSE"],
          [["LSTM", "0.6531", "19.570"],
           ["GRU", "0.6424", "19.832"],
           ["TCN", "0.5969", "21.084"],
           ["Multi-scale CNN", "0.5325", "22.712"]],
          [1.40, 0.85, 1.03], size=9, row_height=0.28)
    textbox(slide, x, CONTENT_TOP + 1.85, width, 1.70,
            ["Its own frozen sampling policy and lookback 20, so these values "
             "are not on a common scale with the left panel. The gate it had to "
             "clear was about R² 0.89 and RMSE 10.7.",
             "State the finding for the architectures, representations and "
             "budgets tested here — not as a claim about deep learning and RUL "
             "in general."],
            size=9, color=DARK)
    source_label(slide, item["source"])

    # --- B3 cap/scenario matrix
    item = items["B3"]
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      "Use if challenged on the claim that the 2x2 cells are "
                      "not on a common scale, or on the difference between "
                      "fitting cap, evaluation support and prediction "
                      "clipping.")
    table(slide, L, CONTENT_TOP + 0.08, W,
          ["scenarios", "fitting target", "model", "inner RMSE", "fold SD",
           "inner R²", "bias", "overpred. rate"],
          [["current", "raw", "XGBoost", "29.8059", "1.7513", "0.7708", "+5.2165", "67.35%"],
           ["current", "raw", "ExtraTrees", "32.7578", "2.6545", "0.7264", "+4.4906", "66.00%"],
           ["current", "cap 125", "XGBoost", "36.2141", "2.5645", "0.6690", "-10.8488", "46.70%"],
           ["current", "cap 125", "ExtraTrees", "37.1100", "2.5884", "0.6525", "-10.7370", "50.95%"],
           ["early/middle", "raw", "XGBoost", "23.3667", "1.5326", "0.4867", "+11.4973", "71.35%"],
           ["early/middle", "raw", "ExtraTrees", "26.6406", "2.9202", "0.2703", "+10.6021", "67.60%"],
           ["early/middle", "cap 125", "XGBoost", "11.9746", "0.9418", "0.8661", "-0.9007", "47.30%"],
           ["early/middle", "cap 125", "ExtraTrees", "12.2693", "0.8427", "0.8598", "-1.0864", "47.15%"]],
          [1.30, 1.10, 1.12, 1.02, 0.78, 0.86, 0.92, 1.92], size=9,
          highlight={6, 7})
    textbox(slide, L, CONTENT_TOP + 2.40, W, 1.00, [item["note"]], size=10,
            color=DARK)
    source_label(slide, item["source"])

    # --- B4 residual correction, nested calibration, safety table
    item = items["B4"]
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      "Use if asked how the residual head avoids leaking, or "
                      "what the calibration cost.")
    headed_box(slide, L, CONTENT_TOP + 0.08, 4.30, 3.30,
               "What the Run 7 adapter does", [
                   "· Splits training UAVs internally into four groups.",
                   "· Fits the six tree members without the held group.",
                   "· Predicts it; repeats, so no UAV is seen by its own model.",
                   "· Filters calibration endpoints to training UAVs and "
                   "balances weight per UAV.",
                   "· Fits the residual head there, then predicts the external "
                   "held-out UAVs once.",
                   "· The earlier PE_11 report is not a fully nested estimate "
                   "of its own stack; the Run 7 adapter does not share that "
                   "dependency.",
                   "· Its remaining limitation: configurations and the workflow "
                   "were chosen on development data now examined many times.",
               ], size=9.5)
    x = L + 4.30 + 0.22
    width = R - x
    textbox(slide, x, CONTENT_TOP + 0.06, width, 0.22,
            ["Conditional calibration policies (PE_4)"], size=10.5, bold=True,
            color=BLUE)
    table(slide, x, CONTENT_TOP + 0.34, width,
          ["policy", "mean-fold R²", "RMSE", "RMS overpred."],
          [["control", "0.89422", "10.5306", "7.255"],
           ["q = 0.50", "0.89456", "10.5354", "6.817"],
           ["q = 0.55", "0.89253", "10.6430", "6.567"],
           ["q = 0.60", "0.88931", "10.8144", "6.236"],
           ["q = 0.65", "0.88106", "11.2152", "5.629"],
           ["q = 0.70", "0.87070", "11.6989", "5.053"]],
          [1.10, 1.10, 0.95, 1.35], size=9, row_height=0.26, highlight={2})
    textbox(slide, x, CONTENT_TOP + 2.30, width, 1.10,
            ["Rule declared in advance: only policies within 0.005 mean-fold R² "
             "of the best are eligible; among those, minimise RMS "
             "overprediction, then overprediction rate, then RMSE.",
             "A later uncertainty-scaled variant improved near-failure RMS "
             "overprediction by 11.31% but raised RMSE by 1.71% and was "
             "rejected."],
            size=9, color=DARK)
    source_label(slide, item["source"])

    # --- B5 rejected representations
    item = items["B5"]
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      "Use if asked what else was tried on the feature side. "
                      "Every row is a matched development comparison whose "
                      "control was retained.")
    table(slide, L, CONTENT_TOP + 0.10, W,
          ["representation tested", "matched development result", "decision"],
          [["Signal compression (median / PCA indices)",
            "R² fell by roughly 0.09-0.11 in both families",
            "keep individual features"],
           ["Dense training prefixes (stride 5)",
            "R² -0.0338 XGBoost, -0.0362 ExtraTrees; lost every fold",
            "keep 20 test-like cutoffs"],
           ["Per-UAV robust normalisation",
            "combined set: +0.0051 XGBoost, -0.0065 ExtraTrees",
            "keep raw features"],
           ["Failure-cycle target instead of RUL",
            "ΔR² -0.4002 XGBoost, -0.5073 ExtraTrees; 0/5 folds",
            "keep RUL target"],
           ["Personalised degradation-onset targets",
            "XGBoost RMSE 12.15 → 19.59-19.69", "keep cap 125"],
           ["Drift-feature pruning",
            "domain AUC 0.8565, but no treatment passed the accuracy gate",
            "keep all features"],
           ["Population degradation features",
            "mean RMSE +2.5% (HGB) and +11.8% (ridge)", "keep control"]],
          [3.05, 4.10, 1.87], size=9.5)
    textbox(slide, L, CONTENT_TOP + 2.20, W, 1.20,
            ["PE_28 now revisits the representation question deliberately: ten "
             "feature sets from 50 to 310 features crossed with two model "
             "recipes, varying sensor coverage, summary complexity, baseline "
             "averaging and window length one factor at a time. Screening uses "
             "grouped seed 20270107; confirmation uses seeds 20270117 and "
             "20270127 on the same UAV population, so it is a robustness check "
             "rather than an untouched holdout."],
            size=10, color=DARK)
    source_label(slide, item["source"])

    # --- B6 uncertainty and provenance
    item = items["B6"]
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      "Use for methodological questions about the intervals, "
                      "the repeated endpoints, or where the public scores come "
                      "from.")
    columns = [
        ("What the bootstrap covers",
         ["1,000-3,000 resamples of whole UAVs, never of rows.",
          "Run 7 pooled R² interval 0.87779 - 0.92772.",
          "Run 7 minus Run 6 RMSE interval -1.252 to +0.446 cycles.",
          "It resamples fixed saved predictions, so it excludes training and "
          "model-selection uncertainty.",
          "A positive mean with an interval crossing zero is uncertain "
          "evidence, not proof of no effect."]),
        ("Repeated endpoints",
         ["500 development rows contain 444 distinct (UAV, cutoff) pairs.",
          "Removing duplicates changes pooled R² only from 0.90446 to 0.90494.",
          "Duplication therefore does not explain the headline number, but it "
          "does affect weighting and precision.",
          "The UAV, not the endpoint, stays the resampling unit."]),
        ("Public-score provenance",
         ["Public scores were transcribed from user-supplied Kaggle "
          "screenshots recorded on 7 September 2026, not from a live lookup.",
          "The public leaderboard uses about 30% of the test data; the final "
          "standing uses the other 70%.",
          "The 0.87877 figure for the alternative script is user-reported and "
          "has no submission artifact in this repository."]),
    ]
    column_width = (W - 0.36) / 3
    for index, (head, lines) in enumerate(columns):
        x = L + index * (column_width + 0.18)
        headed_box(slide, x, CONTENT_TOP + 0.10, column_width, 3.34, head,
                   [f"· {line}" for line in lines], size=9.5)
    source_label(slide, item["source"])


BUILDERS = {
    "title": slide_title,
    "figure": slide_figure,
    "split_diagram": slide_split_diagram,
    "ensemble": slide_ensemble,
    "screen": slide_screen,
    "table": slide_table,
    "conclusions": slide_conclusions,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    prs = Presentation(args.template)
    for layout in prs.slide_masters[0].slide_layouts:
        LAYOUT[layout.name] = layout

    slide_ids = prs.slides._sldIdLst
    for slide_id in list(slide_ids):
        relationship = slide_id.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships}id")
        prs.part.drop_rel(relationship)
        slide_ids.remove(slide_id)

    for item in C.MAIN:
        BUILDERS[item["layout"]](prs, item)
    divider(prs)
    build_backups(prs)

    prs.save(args.out)
    print(f"wrote {args.out} with {len(prs.slides._sldIdLst)} slides")


if __name__ == "__main__":
    main()
