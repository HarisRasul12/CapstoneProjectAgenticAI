from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from textwrap import wrap


PAGE_W = 612
PAGE_H = 792
MARGIN = 48

NAVY = (0.06, 0.09, 0.13)
INK = (0.11, 0.15, 0.20)
MUTED = (0.38, 0.43, 0.49)
BLUE = (0.08, 0.45, 0.78)
CYAN = (0.07, 0.73, 0.88)
PALE = (0.94, 0.97, 0.99)
LINE = (0.82, 0.86, 0.90)
GREEN = (0.10, 0.55, 0.32)
AMBER = (0.82, 0.49, 0.08)


def esc(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", " ")
    )


def rgb(color: tuple[float, float, float]) -> str:
    return f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f}"


@dataclass
class Canvas:
    ops: list[str] = field(default_factory=list)

    def rect(self, x: float, y: float, w: float, h: float, color: tuple[float, float, float]) -> None:
        self.ops.append(f"{rgb(color)} rg {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")

    def stroke_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        color: tuple[float, float, float] = LINE,
        width: float = 0.8,
    ) -> None:
        self.ops.append(f"{width:.2f} w {rgb(color)} RG {x:.2f} {y:.2f} {w:.2f} {h:.2f} re S")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: tuple[float, float, float] = LINE,
        width: float = 0.8,
    ) -> None:
        self.ops.append(
            f"{width:.2f} w {rgb(color)} RG {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S"
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 10,
        font: str = "F1",
        color: tuple[float, float, float] = INK,
    ) -> None:
        self.ops.append(
            f"BT /{font} {size:.2f} Tf {rgb(color)} rg {x:.2f} {y:.2f} Td ({esc(text)}) Tj ET"
        )

    def paragraph(
        self,
        x: float,
        y: float,
        width: float,
        text: str,
        size: float = 9.2,
        leading: float = 12.5,
        font: str = "F1",
        color: tuple[float, float, float] = INK,
    ) -> float:
        max_chars = max(24, int(width / (size * 0.49)))
        for line in wrap(text, width=max_chars):
            self.text(x, y, line, size=size, font=font, color=color)
            y -= leading
        return y

    def bullets(
        self,
        x: float,
        y: float,
        width: float,
        items: list[str],
        size: float = 8.8,
        leading: float = 11.2,
    ) -> float:
        max_chars = max(24, int((width - 12) / (size * 0.49)))
        for item in items:
            lines = wrap(item, width=max_chars)
            self.text(x, y, "-", size=size, font="F2", color=BLUE)
            self.text(x + 12, y, lines[0], size=size, color=INK)
            y -= leading
            for continuation in lines[1:]:
                self.text(x + 12, y, continuation, size=size, color=INK)
                y -= leading
            y -= 1.5
        return y

    def section_label(self, x: float, y: float, label: str) -> None:
        self.rect(x, y - 13, 5, 17, CYAN)
        self.text(x + 12, y - 8, label.upper(), size=10.5, font="F2", color=INK)

    def card(self, x: float, y: float, w: float, h: float, title: str, body: str) -> None:
        self.rect(x, y - h, w, h, PALE)
        self.stroke_rect(x, y - h, w, h, LINE)
        self.text(x + 12, y - 20, title, size=10, font="F2", color=BLUE)
        self.paragraph(x + 12, y - 36, w - 24, body, size=8.5, leading=11.2)

    def stream(self) -> bytes:
        return ("\n".join(self.ops) + "\n").encode("latin-1")


class PDF:
    def __init__(self) -> None:
        self.pages: list[bytes] = []

    def add_page(self, canvas: Canvas) -> None:
        self.pages.append(canvas.stream())

    def write(self, path: Path) -> None:
        objects: list[bytes] = []

        def obj(data: str | bytes) -> int:
            if isinstance(data, str):
                data = data.encode("latin-1")
            objects.append(data)
            return len(objects)

        catalog_id = obj("placeholder")
        pages_id = obj("placeholder")
        font_regular = obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        font_bold = obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        font_italic = obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>")

        page_ids: list[int] = []
        for stream in self.pages:
            content_id = obj(
                b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream"
            )
            page_id = obj(
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
                f"/Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R /F3 {font_italic} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            )
            page_ids.append(page_id)

        objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1")
        kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
        objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode(
            "latin-1"
        )

        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for idx, data in enumerate(objects, start=1):
            offsets.append(len(output))
            output.extend(f"{idx} 0 obj\n".encode("ascii"))
            output.extend(data)
            output.extend(b"\nendobj\n")
        xref = len(output)
        output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
                "ascii"
            )
        )
        path.write_bytes(output)


def footer(c: Canvas, page: int) -> None:
    c.line(MARGIN, 34, PAGE_W - MARGIN, 34, LINE)
    c.text(MARGIN, 20, "ExecLab AI Business Document", size=7.8, color=MUTED)
    c.text(PAGE_W - 74, 20, f"Page {page}", size=7.8, color=MUTED)


def build_page_one() -> Canvas:
    c = Canvas()
    c.rect(0, PAGE_H - 150, PAGE_W, 150, NAVY)
    c.rect(0, PAGE_H - 150, 8, 150, CYAN)
    c.text(MARGIN, PAGE_H - 58, "ExecLab AI", size=31, font="F2", color=(1, 1, 1))
    c.text(
        MARGIN,
        PAGE_H - 82,
        "Agentic execution research lab for benchmark schedules and explainable TCA",
        size=12.5,
        color=(0.83, 0.90, 0.96),
    )
    c.text(
        MARGIN,
        PAGE_H - 111,
        "Live demo: https://execlab-ai-q7smatrnpa-uc.a.run.app",
        size=10,
        font="F2",
        color=CYAN,
    )

    y = PAGE_H - 185
    c.section_label(MARGIN, y, "Business case")
    y -= 28
    y = c.paragraph(
        MARGIN,
        y,
        PAGE_W - 2 * MARGIN,
        "ExecLab AI is an agentic execution-analysis copilot for small funds, student investment funds, fintech product teams, and quant students who need to compare benchmark execution approaches and explain tradeoffs without buying a full institutional OMS/EMS analytics stack.",
        size=10.2,
        leading=14,
    )

    c.card(
        MARGIN,
        y - 16,
        158,
        88,
        "Primary user",
        "Small-fund analysts, student investment fund traders, junior execution traders, quant students, and fintech product teams.",
    )
    c.card(
        MARGIN + 176,
        y - 16,
        158,
        88,
        "Job to be done",
        "Compare execution schedules, understand costs, and produce a defensible memo from live intraday data.",
    )
    c.card(
        MARGIN + 352,
        y - 16,
        164,
        88,
        "Agentic value",
        "Specialist agents turn tool-grounded metrics into explanation, debate, critique, and custom plan design.",
    )

    y -= 135
    c.section_label(MARGIN, y, "The problem")
    y -= 28
    c.bullets(
        MARGIN,
        y,
        PAGE_W - 2 * MARGIN,
        [
            "Execution algorithm vocabulary is often black-box: users know TWAP, VWAP, POV, and IS by name, but cannot see schedule mechanics.",
            "Pre-trade reasoning is fragmented across volume curves, volatility, spread proxies, beta risk, peer movement, and order size.",
            "Metrics alone are not enough; users still need an analyst-quality explanation of why one schedule performed better.",
            "Custom constraints such as max participation or completion-by-time targets require language understanding, not brittle hidden regex parsing.",
        ],
    )

    y = 220
    c.section_label(MARGIN, y, "Product workflow")
    y -= 26
    steps = [
        ("1", "Fetch live intraday bars"),
        ("2", "Run schedule and TCA tools"),
        ("3", "Compress context"),
        ("4", "ADK agents analyze"),
        ("5", "Memo, debate, custom plan"),
    ]
    x = MARGIN
    for number, label in steps:
        c.rect(x, y - 32, 28, 28, BLUE)
        c.text(x + 9, y - 23, number, size=12, font="F2", color=(1, 1, 1))
        c.paragraph(x + 36, y - 12, 72, label, size=8.2, leading=10)
        if number != "5":
            c.line(x + 116, y - 18, x + 132, y - 18, CYAN, 1.4)
        x += 102

    footer(c, 1)
    return c


def build_page_two() -> Canvas:
    c = Canvas()
    c.rect(0, PAGE_H - 62, PAGE_W, 62, NAVY)
    c.text(MARGIN, PAGE_H - 39, "Agentic AI Solution", size=20, font="F2", color=(1, 1, 1))
    c.text(MARGIN, PAGE_H - 54, "Designed around ADK handoff, structured outputs, and tool-grounded reasoning", size=8.5, color=(0.83, 0.90, 0.96))

    y = PAGE_H - 96
    c.section_label(MARGIN, y, "Technical architecture")
    y -= 28
    c.paragraph(
        MARGIN,
        y,
        PAGE_W - 2 * MARGIN,
        "The system separates deterministic tools from agentic interpretation. Python tools fetch data, build curves, simulate fills, calculate TCA, and fit expected-cost models. The service compresses those outputs into structured context. Google ADK agents then explain, debate, critique, and recommend from that evidence.",
        size=9.5,
        leading=13,
    )

    y -= 78
    c.card(
        MARGIN,
        y,
        160,
        96,
        "Tool layer",
        "Data fetch, schedule generation, fill simulation, TCA metrics, expected-cost model, beta risk, peer clustering, scenario lab.",
    )
    c.card(
        MARGIN + 178,
        y,
        160,
        96,
        "Context layer",
        "Raw bars are compressed into benchmark prices, curves, summaries, risk splits, caveats, and chart-ready diagnostics.",
    )
    c.card(
        MARGIN + 356,
        y,
        160,
        96,
        "Agent layer",
        "ADK LlmAgents produce tab commentary, cause-effect TCA, debate, custom plan interpretation, memo, and critic review.",
    )

    y -= 134
    c.section_label(MARGIN, y, "Core agents")
    y -= 27
    left = [
        "MarketDataAgent and VolumeCurveAgent inspect data coverage and volume shape.",
        "PreTradeAnalyticsAgent and ExpectedCostModelAgent explain cost drivers before execution.",
        "BetaRiskMappingAgent and PeerClusterAgent separate market/sector risk from stock-specific pressure.",
        "FastExecutionAdvocate and LiquiditySeekingAdvocate debate opposite trading philosophies.",
    ]
    right = [
        "CustomAlgoPlannerAgent turns natural-language constraints into a structured CustomAlgoPlan.",
        "CustomAlgoDesignerAgent explains the hybrid schedule produced from that plan.",
        "TabInsightAgent writes section-specific commentary across the Streamlit interface.",
        "NarrativeExplanationAgent and CriticGoldenSetAgent produce and review the final memo.",
    ]
    c.bullets(MARGIN, y, 245, left, size=8.4, leading=10.8)
    c.bullets(MARGIN + 270, y, 245, right, size=8.4, leading=10.8)

    y = 260
    c.section_label(MARGIN, y, "Why this is agentic")
    y -= 27
    c.bullets(
        MARGIN,
        y,
        PAGE_W - 2 * MARGIN,
        [
            "The app uses Google ADK SequentialAgent handoff rather than a single prompt.",
            "Each agent has an output schema and writes structured state for the app to consume.",
            "The custom algo chat relies on CustomAlgoPlannerAgent output, not hidden deterministic text parsing.",
            "A critic agent and golden tests check grounding, caveats, and assignment-safe limitation language.",
            "Cloud Run smoke tests verify both planner-level ADK and the full multi-agent synthesis path online.",
        ],
        size=8.7,
        leading=11,
    )

    footer(c, 2)
    return c


def build_page_three() -> Canvas:
    c = Canvas()
    c.rect(0, PAGE_H - 62, PAGE_W, 62, NAVY)
    c.text(MARGIN, PAGE_H - 39, "Economics, Requirements, and Deployment Proof", size=18, font="F2", color=(1, 1, 1))

    y = PAGE_H - 96
    c.section_label(MARGIN, y, "Economics")
    y -= 28
    c.bullets(
        MARGIN,
        y,
        PAGE_W - 2 * MARGIN,
        [
            "Free/student tier: limited number of backtests per month for education and demos.",
            "Student/pro tier: about $10-$49 per month for more runs, saved scenarios, and richer memos.",
            "Team tier: about $199 per month for shared workspaces, templates, and training usage.",
            "Cost to serve is low because calculations are Python, Cloud Run can scale down, and Gemini receives compact structured context rather than raw bars.",
            "The business should be positioned as an execution research and education lab, not a production trading system.",
        ],
        size=8.8,
        leading=11,
    )

    y = 525
    c.section_label(MARGIN, y, "Capstone requirements")
    y -= 30
    rows = [
        ("Agent framework", "Google ADK LlmAgent plus SequentialAgent with Vertex Gemini."),
        ("Deployed URL", "Public Cloud Run app at execlab-ai-q7smatrnpa-uc.a.run.app."),
        ("Original product", "New execution research lab, not a refactor of Project 1 or 2."),
        ("3+ class concepts", "Handoff, tool calling, context engineering, few-shot prompting, golden evals."),
        ("Live demo ready", "Streamlit UI, live data fetch, ADK reports, agent memo, trace tab."),
        ("Business document", "This PDF plus README business-document section."),
    ]
    row_h = 34
    c.rect(MARGIN, y - 4, PAGE_W - 2 * MARGIN, 22, BLUE)
    c.text(MARGIN + 10, y + 2, "Requirement", size=8.5, font="F2", color=(1, 1, 1))
    c.text(MARGIN + 160, y + 2, "Evidence", size=8.5, font="F2", color=(1, 1, 1))
    y -= 18
    for idx, (req, evidence) in enumerate(rows):
        fill = PALE if idx % 2 == 0 else (1, 1, 1)
        c.rect(MARGIN, y - row_h + 6, PAGE_W - 2 * MARGIN, row_h, fill)
        c.stroke_rect(MARGIN, y - row_h + 6, PAGE_W - 2 * MARGIN, row_h, LINE, 0.4)
        c.text(MARGIN + 10, y - 7, req, size=8.5, font="F2", color=INK)
        c.paragraph(MARGIN + 160, y - 7, PAGE_W - 2 * MARGIN - 170, evidence, size=8.1, leading=9.2)
        y -= row_h

    y = 230
    c.section_label(MARGIN, y, "Deployment proof")
    y -= 28
    c.card(
        MARGIN,
        y,
        248,
        90,
        "Cloud Run",
        "Service execlab-ai, region us-central1, verified revision execlab-ai-00005-b5j, public URL HTTP 200.",
    )
    c.card(
        MARGIN + 268,
        y,
        248,
        90,
        "ADK smoke",
        "Full deployed smoke returned ADK_FULL_SMOKE_OK with 19 agent reports using gemini-2.5-flash-lite.",
    )

    y -= 120
    c.section_label(MARGIN, y, "Important limitation")
    y -= 27
    c.paragraph(
        MARGIN,
        y,
        PAGE_W - 2 * MARGIN,
        "ExecLab AI is a bar-based execution research simulator. Public OHLCV bars cannot model queue position, venue routing, hidden liquidity, true NBBO spread capture, or tick-level adverse selection. This limitation is stated in the UI and agent memo.",
        size=8.8,
        leading=11.5,
        font="F3",
        color=MUTED,
    )

    footer(c, 3)
    return c


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "docs" / "ExecLab_AI_Business_Document.pdf"
    pdf = PDF()
    pdf.add_page(build_page_one())
    pdf.add_page(build_page_two())
    pdf.add_page(build_page_three())
    pdf.write(out)
    print(out)


if __name__ == "__main__":
    main()
