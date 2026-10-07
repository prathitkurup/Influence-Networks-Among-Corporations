"""
Read rbo_directed_influence.gml and write a filtered, colored GEXF for Gephi.

Removes balanced edges (net_temporal == 0) and decisive edges below median weight;
drops isolated nodes. Node colors use a red→yellow→green diverging map on net_strength;
edge colors match target node.
Output: visualizations/gexf/rbo_directed_influence.gexf
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx

sys.path.insert(0, ".")
from config import ROOT

GML_IN  = ROOT / "visualizations" / "gml"  / "rbo_directed_influence.gml"
GEXF_OUT = ROOT / "visualizations" / "gexf" / "rbo_directed_influence.gexf"

GEXF_NS = "http://gexf.net/1.3"
VIZ_NS  = "http://gexf.net/1.3/viz"

# -- Color anchors (manual RGB; no matplotlib) --------------------------------
RED    = (214,  39,  40)   # Tableau red
YELLOW = (255, 215,   0)   # gold midpoint
GREEN  = ( 44, 160,  44)   # Tableau green
GRAY   = (150, 150, 150)   # neutral color

# -- Attribute schemas (hardcoded for clarity; types verified against GML) ----
#    (title, gexf_type, id_string)
NODE_ATTR_SCHEMA = [
    ("out_strength",        "float",   "n0"),
    ("in_strength",         "float",   "n1"),
    ("net_strength",        "float",   "n2"),
    ("total_firsts",        "integer", "n3"),
    ("total_losses",        "integer", "n4"),
    ("net_influence",       "integer", "n5"),
    ("color",               "string",  "n6"),   # updated hex after remapping
    ("num_bills",           "integer", "n7"),
    ("bill_aff_community",  "integer", "n8"),
    ("within_comm_net_str", "float",   "n9"),
    ("within_comm_net_inf", "integer", "n10"),
    ("net_weight",          "float",   "n11"),
    ("total_spend",         "float",   "n12"),
]
# weight is the native GEXF edge attribute; remaining attrs go into attvalues
EDGE_ATTR_SCHEMA = [
    ("rbo",           "float",   "e0"),
    ("source_firsts", "integer", "e1"),
    ("target_firsts", "integer", "e2"),
    ("tie_count",     "integer", "e3"),
    ("shared_bills",  "integer", "e4"),
    ("net_temporal",  "integer", "e5"),
]


# ---------------------------------------------------------------------------
# Precomputation (before filtering)
# ---------------------------------------------------------------------------

def compute_net_weight(G):
    """Compute net_weight = in_weighted_degree + out_weighted_degree for each node."""
    for node in G.nodes():
        in_w = sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
        out_w = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
        G.nodes[node]["net_weight"] = in_w + out_w


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def remove_balanced_edges(G):
    """Remove balanced edges (net_temporal == 0); returns count removed."""
    to_drop = [(u, v) for u, v, d in G.edges(data=True) if d.get("net_temporal", 0) == 0]
    G.remove_edges_from(to_drop)
    return len(to_drop)


def filter_directed_below_median(G):
    """
    Remove decisive edges below the median weight of the current edge set.
    Median uses the lower value for even-length lists (conservative).
    Returns (n_edges_removed, median_weight).
    """
    weights = sorted(d["weight"] for _, _, d in G.edges(data=True))
    n = len(weights)
    if n == 0:
        return 0, 0.0
    median_w = weights[(n - 1) // 2]
    to_drop = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < median_w]
    G.remove_edges_from(to_drop)
    return len(to_drop), median_w


def remove_zero_strength_nodes(G):
    """
    Remove nodes where both in_strength and out_strength are 0 in the
    filtered graph (no remaining edges after earlier filtering steps).
    Nodes with equal but nonzero in/out strength are kept.

    Returns count removed.
    """
    to_drop = [
        n for n in G.nodes()
        if (sum(d["weight"] for _, _, d in G.out_edges(n, data=True)) == 0.0
            and sum(d["weight"] for _, _, d in G.in_edges(n, data=True)) == 0.0)
    ]
    G.remove_nodes_from(to_drop)
    return len(to_drop)


# ---------------------------------------------------------------------------
# Color helpers (no matplotlib)
# ---------------------------------------------------------------------------

def _lerp(c1, c2, t):
    """Linearly interpolate between two RGB tuples; t clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def diverging_rgb(value, vmin, vmax):
    """
    Red→yellow→green diverging colormap centered at 0.

    Negative values map RED(vmin) → YELLOW(0).
    Positive values map YELLOW(0) → GREEN(vmax).
    Returns (r, g, b) ints in [0, 255].
    """
    if value <= 0:
        denom = abs(vmin) if vmin != 0 else 1.0
        t = (value - vmin) / denom    # 0 at vmin, 1 at 0
        return _lerp(RED, YELLOW, t)
    else:
        denom = vmax if vmax != 0 else 1.0
        t = value / denom             # 0 at 0, 1 at vmax
        return _lerp(YELLOW, GREEN, t)


def rgb_to_hex(r, g, b):
    return f"#{r:02X}{g:02X}{b:02X}"


def build_node_colors(G):
    """
    Compute diverging colors keyed by node; also update G nodes['color'] attr.
    Returns {node: (r, g, b)}.
    """
    ns_vals = {n: G.nodes[n]["net_strength"] for n in G.nodes()}
    vmin = min(ns_vals.values())
    vmax = max(ns_vals.values())
    colors = {}
    for node, ns in ns_vals.items():
        rgb = diverging_rgb(ns, vmin, vmax)
        colors[node] = rgb
        G.nodes[node]["color"] = rgb_to_hex(*rgb)  # update string attr too
    return colors, vmin, vmax


# ---------------------------------------------------------------------------
# GEXF XML construction
# ---------------------------------------------------------------------------

def _el(parent, tag, attrib=None, text=None):
    """Create a child element in the GEXF namespace."""
    el = ET.SubElement(parent, f"{{{GEXF_NS}}}{tag}", attrib or {})
    if text is not None:
        el.text = text
    return el


def _viz(parent, tag, attrib):
    """Create a child element in the viz namespace."""
    el = ET.SubElement(parent, f"{{{VIZ_NS}}}{tag}", attrib)
    return el


def _attvalue(parent, for_id, value):
    """Append a single <attvalue for=... value=.../> element."""
    ET.SubElement(
        parent, f"{{{GEXF_NS}}}attvalue",
        {"for": for_id, "value": str(value)},
    )


def build_gexf(G, node_colors):
    """
    Construct and return the root ET.Element for the GEXF document.
    """
    ET.register_namespace("",    GEXF_NS)
    ET.register_namespace("viz", VIZ_NS)

    root = ET.Element(f"{{{GEXF_NS}}}gexf", {"version": "1.3"})

    # meta
    meta = _el(root, "meta", {"lastmodifieddate": "2026-04-13"})
    _el(meta, "creator", text="gephi_style_export.py")
    _el(meta, "description",
        text="RBO Directed Influence Network — Fortune 500, 116th Congress")

    # Omit defaultedgetype entirely: Gephi infers a mixed graph from the
    # individual edge type attributes (directed/undirected) without needing
    # a graph-level declaration. Setting defaultedgetype="directed" drops all
    # undirected edges; setting it to "mixed" triggers a SEVERE "not
    # recognized" warning because "mixed" is not a valid GEXF enum value.
    graph = _el(root, "graph", {"mode": "static"})

    # -- attribute declarations: nodes
    node_attrs_el = _el(graph, "attributes", {"class": "node", "mode": "static"})
    node_id_map = {}   # title -> id_string
    for title, gtype, aid in NODE_ATTR_SCHEMA:
        _el(node_attrs_el, "attribute", {"id": aid, "title": title, "type": gtype})
        node_id_map[title] = aid

    # -- attribute declarations: edges
    edge_attrs_el = _el(graph, "attributes", {"class": "edge", "mode": "static"})
    edge_id_map = {}   # title -> id_string
    for title, gtype, aid in EDGE_ATTR_SCHEMA:
        _el(edge_attrs_el, "attribute", {"id": aid, "title": title, "type": gtype})
        edge_id_map[title] = aid

    # -- nodes
    nodes_el = _el(graph, "nodes")
    for node, attrs in sorted(G.nodes(data=True)):   # sorted for determinism
        node_el = _el(nodes_el, "node", {"id": str(node), "label": str(node)})

        attvals = _el(node_el, "attvalues")
        for title, _, aid in NODE_ATTR_SCHEMA:
            val = attrs.get(title)
            if val is not None:
                _attvalue(attvals, aid, val)

        # viz:color from diverging colormap
        r, g, b = node_colors[node]
        _viz(node_el, "color", {"r": str(r), "g": str(g), "b": str(b), "a": "1.0"})

    # -- edges (all directed after balanced removal)
    edges_el = _el(graph, "edges")
    for eid, (src, tgt, edata) in enumerate(
        sorted(G.edges(data=True), key=lambda e: (e[0], e[1]))  # deterministic
    ):
        edge_el = _el(edges_el, "edge", {
            "id":     str(eid),
            "source": str(src),
            "target": str(tgt),
            "weight": str(round(edata["weight"], 6)),
            "type":   "directed",
        })

        # Non-weight edge attributes as attvalues
        attvals = _el(edge_el, "attvalues")
        for title, _, aid in EDGE_ATTR_SCHEMA:
            val = edata.get(title)
            if val is not None:
                _attvalue(attvals, aid, val)

        # viz:color = target node color
        ec = node_colors.get(tgt, (150, 150, 150))
        _viz(edge_el, "color",
             {"r": str(ec[0]), "g": str(ec[1]), "b": str(ec[2]), "a": "0.8"})

    return root


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load
    print("Loading GML...")
    G = nx.read_gml(str(GML_IN), label="label")
    print(f"  Input: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Precompute net_weight on original unfiltered network
    compute_net_weight(G)

    # Step 1: drop balanced edges (net_temporal == 0; both directions have equal weight)
    n_bal = remove_balanced_edges(G)
    print(f"  Removed {n_bal} balanced edges  ({G.number_of_edges()} decisive remain)")

    # Step 2: drop directed edges below their median weight
    n_edge_drop, median_w = filter_directed_below_median(G)
    print(f"  Removed {n_edge_drop} directed edges below median weight {median_w:.6f}")

    # Step 3: drop nodes with 0 in_strength and 0 out_strength
    n_zero = remove_zero_strength_nodes(G)
    print(f"  Removed {n_zero} zero-strength nodes  ({G.number_of_nodes()} remain)")
    print(f"  Retained: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Recompute node colors on net_strength
    node_colors, vmin, vmax = build_node_colors(G)
    print(f"  net_strength range: [{vmin:.4f}, {vmax:.4f}]")
    print(f"  Color: RED(vmin) → YELLOW(0) → GREEN(vmax)")

    # Build and write GEXF
    print(f"\nBuilding GEXF...")
    root = build_gexf(G, node_colors)

    GEXF_OUT.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(GEXF_OUT, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

    size_kb = GEXF_OUT.stat().st_size / 1024
    print(f"\n  Written -> {GEXF_OUT.name}  ({size_kb:.1f} KB)")
    print(f"  Nodes:             {G.number_of_nodes()}")
    print(f"  Decisive edges:    {G.number_of_edges()}  (colored by target node)")
    print(f"  Median weight cut: {median_w:.6f}")
    print(f"  net_strength range: [{vmin:.4f}, {vmax:.4f}]  (color anchor)")


if __name__ == "__main__":
    main()
