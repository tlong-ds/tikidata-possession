"""Rebuild the report's vector figures with matplotlib and the draw.io desktop CLI.

Run from the repository root: python docs/figures/build_figures.py
The chart uses the development means in development_results.json.
"""

import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PDF_DIR = ROOT / "submission" / "figures"
INK = "#243746"
BLUE = "#315f8b"
TEAL = "#087f79"


def architecture():
    """Single-checkpoint architecture, matching the selected model configuration."""
    document = ET.Element("mxfile", host="drawio", version="31.0.2")
    page = ET.SubElement(document, "diagram", name="Switch-Gated Possession Model")
    model = ET.SubElement(page, "mxGraphModel", grid="1", gridSize="10", page="0",
                          pageWidth="1500", pageHeight="350", background="#ffffff")
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    next_id = 2

    def box(label, x, y, width, height, neutral=False, text_only=False):
        nonlocal next_id
        identifier = str(next_id)
        next_id += 1
        style = ("rounded=1;arcSize=10;whiteSpace=wrap;html=1;fontFamily=Helvetica;"
                 "fontSize=20;fontColor=#243746;spacing=8;strokeWidth=1.5;")
        style += ("fillColor=#f4f6f8;strokeColor=#8c9ba5;" if neutral else
                  "fillColor=#eaf2f8;strokeColor=#597b9b;")
        if text_only:
            style += "fillColor=none;strokeColor=none;align=left;fontSize=24;fontStyle=1;"
        node = ET.SubElement(root, "mxCell", id=identifier, value=label,
                             style=style, vertex="1", parent="1")
        ET.SubElement(node, "mxGeometry", x=str(x), y=str(y), width=str(width),
                      height=str(height), **{"as": "geometry"})
        return identifier

    def arrow(source, target, exit_y=0.5, entry_y=0.5, bottom=False, dashed=False):
        nonlocal next_id
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
                 "html=1;strokeColor=#526777;strokeWidth=1.7;endArrow=block;endFill=1;")
        if bottom:
            style += "exitX=1;exitY=0.5;entryX=0.5;entryY=1;"
        else:
            style += f"exitX=1;exitY={exit_y};entryX=0;entryY={entry_y};"
        if dashed:
            style += "dashed=1;"
        edge = ET.SubElement(root, "mxCell", id=str(next_id), style=style, edge="1",
                             parent="1", source=source, target=target)
        next_id += 1
        geometry = ET.SubElement(edge, "mxGeometry", relative="1", **{"as": "geometry"})
        if bottom:
            points = ET.SubElement(geometry, "Array", **{"as": "points"})
            ET.SubElement(points, "mxPoint", x="820", y="290")

    box("Switch-Gated Possession Model", 10, 10, 990, 40, text_only=True)
    players = box("Player features\n(B, T, 46, Fp)", 10, 130, 190, 80, neutral=True)
    mlp = box("Shared MLP\nFp → 64 → 64", 250, 130, 170, 80)
    pooling = box("Team pooling\nH / A means + counts\n(B, T, 130)", 470, 120, 210, 100)
    frame = box("Frame features\nBall + gap state\nGeometry + touches", 470, 250, 210, 100, neutral=True)
    fusion = box("Feature fusion\nLinear → 128", 730, 130, 180, 80)
    temporal = box("Bidirectional GRU\n2 layers, 64 per side\n(B, T, 128)", 960, 120, 210, 100)
    possession = box("Possession head\nE: H / A / D", 1230, 10, 250, 80)
    switch = box("Switch head\ns: start of H / A", 1230, 130, 250, 80)
    auxiliary = box("Auxiliary supervision\nTouch · team · ball xy", 1230, 250, 250, 80, neutral=True)
    arrow(players, mlp)
    arrow(mlp, pooling)
    arrow(pooling, fusion)
    arrow(frame, fusion, bottom=True)
    arrow(fusion, temporal)
    arrow(temporal, possession, exit_y=0.2)
    arrow(temporal, switch)
    arrow(temporal, auxiliary, exit_y=0.8, dashed=True)
    destination = HERE / "architecture.drawio"
    ET.indent(document)
    ET.ElementTree(document).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def robustness_chart():
    """Paired means for two ball conditions on the same dev/select realisations."""
    data = json.loads((HERE / "development_results.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(7.2, 2.35))
    fig.subplots_adjust(left=0.33, right=0.96, bottom=0.23, top=0.76)
    labels = []
    for index, row in enumerate(data["models"]):
        y = 2 - index
        axis.plot([row["target_like"], row["plain"]], [y, y], color="#b7c3cc",
                  linewidth=2.3, zorder=1)
        axis.scatter(row["plain"], y, s=48, facecolor="white", edgecolor=BLUE,
                     linewidth=1.7, zorder=3, label="Original ball" if index == 0 else None)
        axis.scatter(row["target_like"], y, s=40, marker="s", color=TEAL,
                     zorder=3, label="Target-like ball" if index == 0 else None)
        axis.annotate(f'{row["plain"]:.4f}', (row["plain"], y), xytext=(0, 9),
                      textcoords="offset points", ha="center", color=BLUE, fontsize=8.5)
        axis.annotate(f'{row["target_like"]:.4f}', (row["target_like"], y), xytext=(0, -15),
                      textcoords="offset points", ha="center", color=TEAL, fontsize=8.5)
        labels.append(row["label"])
    axis.set_yticks([2, 1, 0], labels)
    axis.get_yticklabels()[0].set_weight("bold")
    axis.set_ylim(-0.65, 2.65)
    axis.set_xlim(0.58, 0.68)
    axis.set_xticks([0.58, 0.60, 0.62, 0.64, 0.66, 0.68])
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_xlabel("Development spellF1", labelpad=6)
    axis.grid(axis="x", color="#e4e9ed", linewidth=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#b7c3cc")
    axis.tick_params(axis="y", length=0, pad=10)
    axis.tick_params(axis="x", length=3, color="#8c9ba5")
    fig.legend(loc="upper center", bbox_to_anchor=(0.67, 1.0), ncol=2,
               frameon=False, fontsize=8.5, handletextpad=0.5, columnspacing=1.5)
    for kind in ("svg", "pdf"):
        output = HERE / "robustness.svg" if kind == "svg" else PDF_DIR / "robustness.pdf"
        fig.savefig(output, format=kind, facecolor="white", metadata={"Creator": "TikiData"})
    plt.close(fig)


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    drawing = architecture()
    drawio = shutil.which("drawio") or shutil.which("draw.io")
    if drawio is None:
        local = Path("/Applications/draw.io.app/Contents/MacOS/draw.io")
        drawio = str(local) if local.exists() else None
    if drawio is None:
        raise SystemExit("Install the draw.io desktop CLI to rebuild the architecture diagram.")
    preview = Path(tempfile.mkdtemp(prefix="tikidata-architecture-")) / "architecture.png"
    subprocess.run([drawio, "-x", "-f", "png", "--width", "2000", "-b", "10",
                    "-o", str(preview), str(drawing)], check=True)
    subprocess.run([drawio, "-x", "-f", "pdf", "-e", "-b", "10", "-o",
                    str(PDF_DIR / "architecture.pdf"), str(drawing)], check=True)
    subprocess.run(["pdftocairo", "-svg", str(PDF_DIR / "architecture.pdf"),
                    str(HERE / "architecture.svg")], check=True)
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(PDF_DIR / "pipeline.pdf"),
                    str(HERE / "pipeline.svg")], check=True)
    robustness_chart()
    print("Architecture preview:", preview)
    print("Vector figures rebuilt in docs/figures/ and submission/figures/.")


if __name__ == "__main__":
    main()
