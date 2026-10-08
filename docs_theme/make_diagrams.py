"""Generate the five topology diagrams as theme-aware inline SVG snippets.

Colours come from CSS custom properties in docs_theme/assets/docs.css (--tc is set per
diagram), so the same SVG reads correctly in light and dark themes.
Run: python docs_theme/make_diagrams.py
"""
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "_snippets" / "diagrams"


def arrow(x1, y1, x2, y2, cls="edge", head=True, both=False, size=7.5):
    """A straight edge; arrowheads are drawn as filled triangles."""
    hc = "head-io" if cls == "edge-io" else "head"
    ang = math.atan2(y2 - y1, x2 - x1)
    out = []
    # shorten the line so it ends at the base of the head
    sx2, sy2 = (x2 - size * math.cos(ang), y2 - size * math.sin(ang)) if head else (x2, y2)
    sx1, sy1 = (x1 + size * math.cos(ang), y1 + size * math.sin(ang)) if both else (x1, y1)
    out.append(f'<path class="{cls}" d="M{sx1:.1f} {sy1:.1f} L{sx2:.1f} {sy2:.1f}"/>')

    def tri(tx, ty, a):
        p1 = (tx, ty)
        p2 = (tx - size * math.cos(a) + size * .55 * math.sin(a), ty - size * math.sin(a) - size * .55 * math.cos(a))
        p3 = (tx - size * math.cos(a) - size * .55 * math.sin(a), ty - size * math.sin(a) + size * .55 * math.cos(a))
        return f'<path class="{hc}" d="M{p1[0]:.1f} {p1[1]:.1f} L{p2[0]:.1f} {p2[1]:.1f} L{p3[0]:.1f} {p3[1]:.1f}Z"/>'
    if head:
        out.append(tri(x2, y2, ang))
    if both:
        out.append(tri(x1, y1, ang + math.pi))
    return "".join(out)


def node(cx, cy, w, h, label, sub=None, cls="node", r=12, text_cls=""):
    x, y = cx - w / 2, cy - h / 2
    s = f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="{r}"/>'
    ty = cy + (4.5 if sub is None else -2)
    tc = f' class="{text_cls}"' if text_cls else ""
    s += f'<text{tc} x="{cx:.1f}" y="{ty:.1f}" text-anchor="middle">{label}</text>'
    if sub:
        sc = "t-small t-on" if text_cls == "t-on" else "t-small"
        s += f'<text class="{sc}" x="{cx:.1f}" y="{cy + 13:.1f}" text-anchor="middle">{sub}</text>'
    return s


def io(cx, cy, label, w=74):
    return node(cx, cy, w, 34, label, cls="io", r=17)


def text(x, y, s, cls="t-small", anchor="middle"):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{s}</text>'


def svg(name, h, label, body, top=0):
    return (f'<div class="diagram-wrap"><svg class="topo-diagram" style="--tc: var(--t-{name})" viewBox="0 {top} 620 {h - top}" '
            f'role="img" aria-label="{label}" xmlns="http://www.w3.org/2000/svg">{body}</svg></div>\n')


def single():
    b = []
    b.append(io(52, 120, "Task"))
    b.append(arrow(89, 120, 228, 120, "edge-io"))
    b.append(node(310, 120, 160, 58, "Agent", "reasons and acts", cls="node-strong", text_cls="t-on"))
    b.append(node(310, 30, 120, 36, "Tools"))
    # loop to tools and back
    b.append(arrow(290, 91, 290, 50))
    b.append(arrow(330, 50, 330, 91))
    b.append(text(282, 74, "call", anchor="end"))
    b.append(text(338, 74, "result", anchor="start"))
    b.append(arrow(390, 120, 530, 120, "edge-io"))
    b.append(io(568, 120, "Answer"))
    b.append(text(310, 178, "Loops until the agent replies without a tool call"))
    return svg("single", 192, "Single topology: one agent calls tools in a loop and answers.", "".join(b))


def independent():
    b = []
    b.append(io(46, 130, "Task", w=70))
    ys = [40, 100, 160, 220]
    for i, y in enumerate(ys):
        b.append(arrow(81, 130, 186, y, "edge-io"))
        b.append(node(230, y, 88, 40, f"Agent {i + 1}"))
        b.append(arrow(274, y, 370, 130))
    b.append(node(422, 130, 104, 50, "Aggregate", "vote or best-of-N", cls="node-strong", text_cls="t-on"))
    b.append(arrow(474, 130, 530, 130, "edge-io"))
    b.append(io(568, 130, "Answer"))
    b.append(text(230, 258, "Same task, no messages between agents"))
    return svg("independent", 268, "Independent topology: four agents answer the same task in parallel and their answers are aggregated.", "".join(b))


def sequential():
    b = []
    roles = ["planner", "retriever", "reasoner", "writer"]
    xs = [158, 270, 382, 494]
    b.append(io(46, 90, "Task", w=66))
    b.append(arrow(79, 90, 114, 90, "edge-io"))
    for i, (x, r) in enumerate(zip(xs, roles)):
        b.append(node(x, 90, 88, 52, f"Stage {i + 1}", r, cls="node-strong" if i == 3 else "node",
                      text_cls="t-on" if i == 3 else ""))
        if i < 3:
            b.append(arrow(x + 44, 90, xs[i + 1] - 44, 90))
    # every stage reads all earlier stages: dashed context arcs below
    for a, c, depth in [(0, 2, 38), (1, 3, 38), (0, 3, 62)]:
        x1, x2 = xs[a], xs[c]
        b.append(f'<path class="edge-soft" d="M{x1} 116 C{x1} {116 + depth}, {x2} {116 + depth}, {x2} {121}"/>')
        b.append(f'<path class="head" d="M{x2} 116 L{x2 - 4} 124 L{x2 + 4} 124Z" opacity=".6"/>')
    b.append(arrow(538, 90, 556, 90, "edge-io", size=6))
    b.append(io(588, 90, "Answer", w=60))
    b.append(text(326, 196, "Each stage reads the task and every earlier stage's output · roles shown for HotpotQA"))
    return svg("sequential", 206, "Sequential topology: four stages run in order; each reads all earlier stages; the last stage answers.", "".join(b), top=52)


def centralized():
    b = []
    b.append(io(52, 58, "Task"))
    b.append(arrow(89, 58, 238, 58, "edge-io"))
    b.append(node(310, 58, 140, 54, "Manager", "plans and delegates", cls="node-strong", text_cls="t-on"))
    wx = [190, 310, 430]
    for i, x in enumerate(wx):
        b.append(node(x, 190, 96, 42, f"Worker {i + 1}"))
        sx = 270 + i * 40
        b.append(arrow(sx - 6, 86, x - 8, 168))          # delegate
        b.append(arrow(x + 8, 168, sx + 6, 86))          # report back
    b.append(arrow(380, 58, 530, 58, "edge-io"))
    b.append(io(568, 58, "Answer"))
    b.append(text(216, 118, "delegate ↓", anchor="end"))
    b.append(text(404, 118, "↑ report back", anchor="start"))
    b.append(text(310, 240, "Workers never talk to each other · loops until the manager ends or a turn cap"))
    return svg("centralized", 252, "Centralized topology: a manager delegates to three workers, collects their reports and answers.", "".join(b), top=20)


def decentralized():
    b = []
    b.append(io(46, 130, "Task", w=70))
    peers = {1: (250, 50), 2: (360, 130), 3: (250, 210), 4: (140, 130)}
    # complete graph between peers (round 1 exchange)
    pairs = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4)]
    for a, c in pairs:
        (x1, y1), (x2, y2) = peers[a], peers[c]
        d = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        b.append(arrow(x1 + ux * 30, y1 + uy * 22, x2 - ux * 30, y2 - uy * 22, "edge-soft", both=True, size=6))
    for i, (x, y) in peers.items():
        b.append(node(x, y, 74, 38, f"Peer {i}"))
    b.append(arrow(81, 130, 101, 130, "edge-io", size=6))
    b.append(arrow(397, 130, 452, 130))
    b.append(node(490, 130, 76, 50, "Vote", "final round", cls="node-strong", text_cls="t-on"))
    b.append(arrow(528, 130, 548, 130, "edge-io", size=6))
    b.append(io(584, 130, "Answer", w=64))
    b.append(text(250, 256, "Round 0: each peer answers alone · Round 1: each reads the others' answers and revises"))
    return svg("decentralized", 266, "Decentralized topology: four peers answer, exchange answers for a round, and the final answers are put to a vote.", "".join(b), top=18)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (single, independent, sequential, centralized, decentralized):
        (OUT / f"{fn.__name__}.svg").write_text(fn())
        print("wrote", fn.__name__)
