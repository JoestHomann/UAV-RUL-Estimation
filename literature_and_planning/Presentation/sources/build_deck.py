"""Build UAV_RUL_Project_20min.pptx on a copy of the supplied University of
Stuttgart template.

The template's masters, layouts, theme colours, fonts and logo are reused
unchanged; only the ten sample slides are removed. Every scientific chart is a
figure the repository already generated or produced by re-running an unmodified
repository script (see ``figure_manifest.csv``); the only native PowerPoint
drawings are the nested-split schematic, the feature catalogue, the
retained-model schematic and the tables, because the repository has no figure
for those.

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
FOOTER_DATE = "9 September 2026"

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
            ["University of Stuttgart · 9 September 2026"], size=10,
            color=GREY)
    textbox(slide, 0.30, 4.92, 4.72, 0.20, [item["source"]], size=8,
            color=GREY)
    return slide


def slide_figure(prs, item):
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      notes_text(item))
    place_picture(slide, item["figure"], (L, CONTENT_TOP + 0.02, W, 3.86))
    source_label(slide, item["source"], y=CONTENT_BOTTOM - 0.16)
    return slide


def panel_column(slide, x, y, width, height, panels, size=9.5):
    """Stack the item's headed panels in one column."""
    gap = 0.09
    each = (height - gap * (len(panels) - 1)) / len(panels)
    for index, (head, lines) in enumerate(panels):
        headed_box(slide, x, y + index * (each + gap), width, each, head,
                   [f"· {line}" for line in lines], size=size)


def slide_split_figure(prs, item):
    """One repository figure beside the panel that says what it decided."""
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      notes_text(item))
    figure_width = item.get("figure_width", 5.42)
    place_picture(slide, item["figure"],
                  (L, CONTENT_TOP + 0.02, figure_width, 3.86))
    x = L + figure_width + 0.22
    panel_column(slide, x, CONTENT_TOP + 0.02, R - x, 3.72, item["panels"])
    source_label(slide, item["source"], y=CONTENT_BOTTOM - 0.16)
    return slide


def slide_figure_notes(prs, item):
    """A wide repository figure above three panels."""
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      notes_text(item))
    figure_height = item.get("figure_height", 2.44)
    place_picture(slide, item["figure"],
                  (L, CONTENT_TOP, W, figure_height), top_align=True)
    y = CONTENT_TOP + figure_height + 0.10
    height = CONTENT_BOTTOM - 0.22 - y
    panels = item["panels"]
    gap = 0.18
    width = (W - gap * (len(panels) - 1)) / len(panels)
    for index, (head, lines) in enumerate(panels):
        headed_box(slide, L + index * (width + gap), y, width, height, head,
                   [f"· {line}" for line in lines], size=9)
    source_label(slide, item["source"], y=CONTENT_BOTTOM - 0.16)
    return slide


def slide_phase1_design(prs, item):
    """Phase 1: the nested whole-UAV design and the automated leakage gate.
    The repository stores these as configuration and a verification report,
    not as a figure, so both halves are drawn natively."""
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      notes_text(item))
    left_width = 5.10
    top = CONTENT_TOP + 0.02

    textbox(slide, L, top, left_width, 0.22,
            ["100 training UAVs · 5 outer folds, balanced by terminal lifetime"],
            size=10.5, bold=True, color=BLUE)

    row_y = top + 0.28
    fold_width = (left_width - 4 * 0.06) / 5
    for index in range(5):
        x = L + index * (fold_width + 0.06)
        held = index == 0
        box = rounded(slide, x, row_y, fold_width, 0.44,
                      YELLOW if held else BLUE, shape=MSO_SHAPE.RECTANGLE)
        set_text(box.text_frame, [f"fold {index + 1}", "20 UAVs"], size=8.5,
                 color=DARK if held else WHITE, align=PP_ALIGN.CENTER,
                 space_after=0)
        box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    textbox(slide, L, row_y + 0.48, left_width, 0.20,
            ["Outer round 1: fold 1 held out, folds 2-5 are the 80 training "
             "UAVs. Every UAV is held out exactly once."],
            size=9, color=DARK)

    inner_label_y = row_y + 0.90
    textbox(slide, L, inner_label_y, left_width, 0.22,
            ["Inside those 80 UAVs · 4 inner folds select everything"],
            size=10.5, bold=True, color=BLUE)
    inner_y = inner_label_y + 0.28
    inner_width = (left_width - 3 * 0.06) / 4
    for index in range(4):
        x = L + index * (inner_width + 0.06)
        box = rounded(slide, x, inner_y, inner_width, 0.40,
                      CYAN if index == 0 else LBLUE, shape=MSO_SHAPE.RECTANGLE)
        set_text(box.text_frame, [f"inner {index + 1}", "60 train / 20 val"],
                 size=8, color=DARK, align=PP_ALIGN.CENTER, space_after=0)
        box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    textbox(slide, L, inner_y + 0.44, left_width, 0.20,
            ["Features, hyperparameters, early stopping and blend weights. "
             "5 x 4 = 20 inner rounds."],
            size=9, color=DARK)

    headed_box(slide, L, inner_y + 0.74, left_width, 1.36,
               "Rules that follow from splitting by UAV", [
                   "· Every prefix of one UAV stays inside one fold.",
                   "· Features use only cycles at or before the cutoff.",
                   "· Scalers are refitted inside each training partition.",
                   "· Each UAV carries a total training weight of 1.",
                   "· Uncertainty resamples whole UAVs, never rows.",
               ], size=9.5)

    x = L + left_width + 0.24
    width = R - x
    headed_box(slide, x, top, width, 2.34,
               "Ten assertions re-checked automatically", [
                   "· Outer folds disjoint: 5 x 20 UAVs.",
                   "· Outer-validation UAVs absent from every inner fold.",
                   "· Development and locked cutoffs reproduce the exact test "
                   "history-length multiset.",
                   "· Feature tables finite; no target or future term in any "
                   "feature name.",
                   "· Prefix causality: replacing all post-cutoff telemetry "
                   "with extreme values leaves the prefix features unchanged.",
                   "· Saved scaler parameters match a fresh recomputation.",
                   "· Baseline predictions carry each UAV's own held-out fold.",
               ], size=9)

    status = rounded(slide, x, top + 2.46, width, 0.62, YELLOW)
    set_text(status.text_frame,
             ["verification_report.json · status: passed · "
              "2,000 training rows · 500 development and 2,000 locked "
              "validation rows · 100 test rows · 606 features"],
             size=9.5, color=DARK, space_after=0)
    status.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    source_label(slide, item["source"], y=CONTENT_BOTTOM - 0.16)
    return slide


FEATURE_SETS = [
    ["age_only", "2", "flight cycle and log(1 + cycle)",
     "is age alone predictive?"],
    ["last_values", "24", "age plus the latest value of each channel",
     "does the current snapshot help?"],
    ["screened", "310", "all temporal features of the 10 degradation "
     "channels, 7 level statistics of the 4 context channels",
     "do the Phase 0 roles help?"],
    ["all_nonconstant", "606", "every generated feature of all 22 channels",
     "did screening discard something useful?"],
]


def slide_features(prs, item):
    """Phase 1 steps 5-7. Catalog and preprocessing counts, no figure."""
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      notes_text(item))
    top = CONTENT_TOP + 0.02

    headed_box(slide, L, top, 3.30, 1.94,
               "27 features per channel, from the prefix only", [
                   "· current and first value",
                   "· baseline mean of the first 10 cycles, and the deviation "
                   "from it",
                   "· history mean, SD, minimum, maximum, slope",
                   "· latest change, mean and maximum absolute change",
                   "· mean, SD, slope, net change and latest-value deviation "
                   "over the last 5, 20 and 50 cycles",
               ], size=9)
    headed_box(slide, L, top + 2.04, 3.30, 1.02,
               "22 x 27 + 10 + 2 = 606 candidates", [
                   "· 10 state features for telemetry_07 and 16",
                   "· 2 age features",
                   "· no feature may reference terminal lifetime or any "
                   "post-cutoff cycle",
               ], size=9)

    x = L + 3.30 + 0.24
    width = R - x
    table(slide, x, top, width,
          ["feature set", "count", "contents", "question it answers"],
          FEATURE_SETS, [1.08, 0.57, 2.20, 1.63], size=8,
          row_height=0.46, highlight={2})

    scaling = rounded(slide, x, top + 2.42, width, 0.64, WHITE, line=GREY)
    set_text(scaling.text_frame,
             ["Fold-fitted scaling: value minus the training-fold median, "
              "divided by IQR / 1.349. 4,405 features scaled by IQR, 293 by a "
              "standard-deviation fallback, 12 by unit fallback."],
             size=9, color=DARK, space_after=0)
    scaling.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    note = rounded(slide, x, top + 3.14, width, 0.42, YELLOW)
    set_text(note.text_frame,
             ["The final set is not chosen here. It is chosen on the inner "
              "folds."], size=9.5, color=DARK, space_after=0)
    note.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    source_label(slide, item["source"], y=CONTENT_BOTTOM - 0.16)
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
    picture = place_picture(slide, item["figure"],
                            (L, CONTENT_TOP + 0.10, figure_width, 3.40))
    caption_y = (picture.top + picture.height) / 914400 + 0.10
    textbox(slide, L, caption_y, figure_width, 0.24,
            ["Run 7 development out-of-fold predictions: the error is "
             "systematic, not random"],
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
        "End of the main deck. Nine backup slides follow: the complete Phase 0 "
        "broad review, redundancy and anomaly evidence, the Phase 1 feature "
        "derivation, the cycle-only baseline group by group, the full locked "
        "architecture table, the sequence and hybrid studies, the complete "
        "cap/scenario matrix, residual-correction and calibration detail, and "
        "rejected experiments with score provenance.")
    return slide


def build_backups(prs):
    items = {item["n"]: item for item in C.BACKUP}

    # --- B1 the complete Phase 0 broad review
    item = items["B1"]
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      "Use if asked what else Phase 0 looked at, or why a "
                      "particular preprocessing choice was made.")
    table(slide, L, CONTENT_TOP + 0.06, W,
          ["analysis", "what it showed", "what it decided"],
          [["Cycle-wise fleet trends",
            "6 channels flat across all cycles; late cycles have few surviving "
            "UAVs", "removal shortlist; treat late-cycle trends cautiously"],
           ["Descriptive statistics",
            "channel medians span about 11 to about 50,000",
            "robust scaling for scale-sensitive models; trees need none"],
           ["Tukey extreme-value screen",
            "~97% of extremes in 04/10/11 are isolated single cycles; 01 and 18 "
            "form long runs in 8-9 UAVs",
            "spike flags vs regime flags; bounds fitted inside each fold"],
           ["Row-level statistics",
            "asymmetry concentrated in 07, 16, 18, 23, 25",
            "robust scaling or Yeo-Johnson for linear and neural models"],
           ["UAV-level statistics",
            "01, 06, 12, 18, 26 differ strongly between UAVs",
            "candidates for context and baseline features"],
           ["Age-band statistics",
            "14 channels change monotonically across the 1-50 / 51-100 / "
            "101-200 / >200 bands",
            "prioritised for temporal features; report metrics by band"],
           ["Histograms and box plots",
            "05, 07, 16, 18 are discrete or regime-like",
            "operating-state features rather than smooth sensors"],
           ["Flatline duration",
            "20 and 27 are 100% flatlined; 03/08/14/17 above 92%; 07 at 96.5%",
            "removal for the six; state features for 07 and 16"]],
          [1.86, 3.66, 3.50], size=8, row_height=0.32)
    textbox(slide, L, CONTENT_TOP + 3.06, W, 0.60,
            ["No channel was removed on distribution evidence alone. Skewness, "
             "extremes, flatlines and regimes are reasons to represent a "
             "channel differently; only UAV-grouped validation decides whether "
             "it earns its place."],
            size=9.5, color=DARK)
    source_label(slide, item["source"])

    # --- B2 redundancy and anomalies
    item = items["B2"]
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      "Use if asked why highly correlated channels were kept, "
                      "or whether anomalous rows were deleted. The repository's "
                      "four correlation heat maps are 28 x 28 and are not "
                      "readable at projection size, so the strongest pairs are "
                      "tabulated instead.")
    column = (W - 0.30) / 2
    top = CONTENT_TOP + 0.04
    textbox(slide, L, top, column, 0.22,
            ["Strongest correlated pairs"], size=10.5, bold=True, color=BLUE)
    table(slide, L, top + 0.26, column,
          ["pair", "row r", "UAV r", "group"],
          [["telemetry_19 / 21", "0.963", "0.999", "1"],
           ["telemetry_06 / 12", "-0.839", "-0.999", "4"],
           ["telemetry_13 / 28", "-0.744", "-0.995", "3"],
           ["telemetry_25 / 28", "0.853", "0.993", "3"],
           ["telemetry_16 / 22", "-0.677", "-0.992", "3"],
           ["telemetry_15 / 23", "-0.954", "-0.983", "2"]],
          [1.62, 0.92, 0.92, 0.90], size=9, row_height=0.26)
    textbox(slide, L, top + 2.16, column, 0.80,
            ["Groups: 1 = 19/21 · 2 = 15/23 · 3 = 13/16/22/25/28 · "
             "4 = 06/07/11/12/24.",
             "Fourteen of the twenty-two channels have a peer above 0.90 in at "
             "least one correlation view. None was removed here."],
            size=9, color=DARK)

    x = L + column + 0.30
    textbox(slide, x, top, column, 0.22,
            ["Anomaly diagnostics, training split"], size=10.5, bold=True,
            color=BLUE)
    table(slide, x, top + 0.26, column,
          ["channel", "extreme rows", "UAVs", "jump rows", "UAVs"],
          [["telemetry_26", "8.45%", "48", "2.02%", "8"],
           ["telemetry_18", "7.25%", "9", "19.48%", "84"],
           ["telemetry_01", "5.28%", "8", "0.00%", "0"],
           ["telemetry_24", "4.63%", "99", "3.54%", "99"],
           ["telemetry_10", "1.80%", "99", "3.43%", "99"],
           ["telemetry_04", "1.80%", "99", "3.41%", "99"]],
          [1.16, 1.06, 0.62, 0.90, 0.62], size=9, row_height=0.26)

    box = rounded(slide, x, top + 2.16, column, 1.24, YELLOW)
    set_text(box.text_frame,
             [("Nothing was deleted", 0),
              ("Extremes in 04, 10 and 24 occur in almost every UAV, which "
               "reads as normal operating behaviour. Those in 01 and 18 sit in "
               "8 and 9 UAVs, so they read as UAV-specific regimes. Persistent "
               "shifts appear only in 19 and 21, five UAVs each, and became "
               "candidate change-point features.", 0)],
             size=9, color=DARK, space_after=3)
    box.text_frame.paragraphs[0].runs[0].font.bold = True
    source_label(slide, item["source"])

    # --- B3 prefix feature derivation
    item = items["B3"]
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      "Use if asked exactly what a feature is, or how the "
                      "causality check works.")
    table(slide, L, CONTENT_TOP + 0.06, 5.72,
          ["group", "features", "what it tells the model"],
          [["Age", "c, log(1 + c)", "accumulated operating time"],
           ["Level", "x_1, x_c", "start level and value at prediction time"],
           ["Baseline", "mean(x_1..x_10), x_c − baseline",
            "movement from the UAV's own starting level"],
           ["History", "mean, SD, min, max, OLS slope",
            "typical level, spread and long-term direction"],
           ["Change", "x_c − x_(c−1), mean and max |Δx|",
            "abrupt change and typical volatility"],
           ["Recent 5 / 20 / 50", "mean, SD, slope, net change, deviation",
            "short, medium and longer-term behaviour"],
           ["State (07, 16)", "unique values, transitions, transition rate, "
            "run length, time since change",
            "operating mode, dwell time, flatlining"]],
          [1.16, 2.20, 2.36], size=8.5, row_height=0.36)
    x = L + 5.72 + 0.24
    width = R - x
    headed_box(slide, x, CONTENT_TOP + 0.06, width, 1.50,
               "How causality is enforced", [
                   "· Only cycles 1..cutoff enter any calculation.",
                   "· The raw files are never truncated; the cutoff is applied "
                   "while computing.",
                   "· Feature names may not contain RUL, target, terminal, "
                   "lifetime, final or future.",
               ], size=9)
    headed_box(slide, x, CONTENT_TOP + 1.62, width, 1.86,
               "How it is verified", [
                   "· 10 prefixes are recomputed after every post-cutoff "
                   "telemetry value is replaced by an extreme artificial "
                   "number.",
                   "· The prefix features must come out identical.",
                   "· Repeated for training, development, locked and test "
                   "tables; all four are also checked for missing and "
                   "non-finite values.",
               ], size=9)
    source_label(slide, item["source"])

    # --- B4 cycle-only baseline, group by group
    item = items["B4"]
    slide = new_slide(prs, "Titel und Inhalt", item["title"], item["subtitle"],
                      "Use if asked whether the benchmark is weak only on "
                      "average, or how much it varies across UAV groups.")
    column = (W - 0.30) / 2
    textbox(slide, L, CONTENT_TOP + 0.04, column, 0.22,
            ["By cutoff band"], size=10.5, bold=True, color=BLUE)
    table(slide, L, CONTENT_TOP + 0.30, column,
          ["band", "rows", "R²", "RMSE", "bias"],
          [["1-50", "80", "-0.014", "37.683", "+4.980"],
           ["51-100", "340", "-0.012", "61.567", "+11.320"],
           ["101-200", "1,020", "-0.386", "67.131", "+38.872"],
           [">200", "560", "-0.375", "64.577", "-15.128"]],
          [0.94, 0.72, 0.86, 0.98, 0.86], size=9, row_height=0.28)

    textbox(slide, L, CONTENT_TOP + 1.80, column, 0.22,
            ["By outer fold (20 UAVs each)"], size=10.5, bold=True, color=BLUE)
    table(slide, L, CONTENT_TOP + 2.06, column,
          ["fold", "R²", "RMSE", "MAE", "bias"],
          [["1", "+0.040", "55.570", "46.943", "+30.498"],
           ["2", "+0.032", "57.025", "49.669", "+22.117"],
           ["3", "-0.068", "76.307", "59.860", "+11.177"],
           ["4", "+0.020", "70.106", "53.041", "+5.375"],
           ["5", "-0.055", "61.306", "50.046", "+19.395"]],
          [0.70, 0.86, 0.94, 0.94, 0.92], size=9, row_height=0.26)

    x = L + column + 0.30
    textbox(slide, x, CONTENT_TOP + 0.04, column, 0.22,
            ["By terminal-lifetime quintile"], size=10.5, bold=True, color=BLUE)
    table(slide, x, CONTENT_TOP + 0.30, column,
          ["quintile", "R²", "RMSE", "MAE", "bias"],
          [["1 shortest", "-3.274", "74.727", "72.359", "+72.359"],
           ["2", "-0.783", "56.301", "52.649", "+52.649"],
           ["3", "+0.330", "40.994", "35.108", "+33.089"],
           ["4", "+0.746", "31.938", "26.764", "+1.286"],
           ["5 longest", "-0.059", "96.859", "72.680", "-70.821"]],
          [1.00, 0.78, 0.86, 0.84, 0.88], size=9, row_height=0.26)

    box = rounded(slide, x, CONTENT_TOP + 2.04, column, 1.40, YELLOW)
    set_text(box.text_frame,
             [("What the quintiles show", 0),
              ("A cycle-only fit cannot be right for both short- and "
               "long-lived UAVs at once: it overpredicts the shortest by 72 "
               "cycles and underpredicts the longest by 71. Fold R² alone "
               "ranges from -0.068 to +0.040, which is why a single 80/20 "
               "split would not have been a measurement.", 0)],
             size=9.5, color=DARK, space_after=3)
    box.text_frame.paragraphs[0].runs[0].font.bold = True
    source_label(slide, item["source"])

    # --- B5 full locked architecture comparison
    item = items["B5"]
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

    # --- B6 sequence and hybrid models
    item = items["B6"]
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

    # --- B7 cap/scenario matrix
    item = items["B7"]
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

    # --- B8 residual correction, nested calibration, safety table
    item = items["B8"]
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

    # --- B9 rejected representations
    item = items["B9"]
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

    # --- B10 uncertainty and provenance
    item = items["B10"]
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
    "split_figure": slide_split_figure,
    "figure_notes": slide_figure_notes,
    "phase1_design": slide_phase1_design,
    "features": slide_features,
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
