
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

# ===============================================
# Config
# ===============================================
MIN_COMMUNITY_SIZE = 440          # keep communities with size > this
OUTLIER_SIZE_PERC = (5, 95)       # remove communities with size outside these percentiles (robust trimming)
MAX_COMMUNITIES = None            # e.g., 25 to keep the largest N communities after filtering
DROP_ISOLATES = True              # remove communities with zero inter-community degree after aggregation

SPRING_K = 3.0                    # layout spacing
SPRING_ITERS = 100

# Node size scaling (consistent with your original)
NODE_SIZE_SCALE = 15.0            # sqrt(size) * scale

# Edge styling
EDGE_MIN_LW = 1.2                 # minimum line width
EDGE_MAX_LW = 5.0                 # maximum line width
EDGE_ALPHA_MIN = 0.35             # min alpha for faint edges
EDGE_ALPHA_MAX = 0.95             # max alpha for strong edges
EDGE_WEIGHT_PERC = (5, 95)        # robust normalization window for weights
EDGE_GRADIENT_STEPS = 16          # segments per edge to approximate smooth gradient

# ===============================================
# Load data
# ===============================================
print("Loading data...")
communities_df = pd.read_csv("./louvain_out/communities.csv")
nodes_df = pd.read_csv("./louvain_out/nodes.csv")
edges_df = pd.read_csv("./louvain_out/edges.csv")

# ===============================================
# Filter communities by size + remove outliers
# ===============================================
base = communities_df[communities_df["size"] > MIN_COMMUNITY_SIZE].copy()
if len(base) == 0:
    print(f"No communities > {MIN_COMMUNITY_SIZE}. Max in data: {communities_df['size'].max()}")
    raise SystemExit(0)

# Robust trimming to drop outliers that distort scaling
lo_p, hi_p = OUTLIER_SIZE_PERC
lo = np.percentile(base["size"], lo_p)
hi = np.percentile(base["size"], hi_p)
trimmed = base[(base["size"] >= lo) & (base["size"] <= hi)].copy()

dropped_outliers = set(base["community"]) - set(trimmed["community"])
print(f"Kept after size/outlier filter: {len(trimmed)} (dropped {len(dropped_outliers)} outliers outside {lo_p}-{hi_p}th percentiles)")

# Optionally cap at N largest (post-trimming)
if MAX_COMMUNITIES is not None and MAX_COMMUNITIES > 0 and len(trimmed) > MAX_COMMUNITIES:
    trimmed = trimmed.sort_values("size", ascending=False).head(MAX_COMMUNITIES).copy()
    print(f"Capped to top {MAX_COMMUNITIES} by size.")

# node -> community map
node_to_comm = dict(zip(nodes_df["user_id"], nodes_df["community"]))

# ===============================================
# Build aggregated graph (community-level)
# ===============================================
print("Building aggregated graph...")
G = nx.Graph()
for _, row in trimmed.iterrows():
    G.add_node(row["community"], size=row["size"])

kept_set = set(trimmed["community"].tolist())

# Aggregate inter-community weights
added = 0
for _, e in edges_df.iterrows():
    u_comm = node_to_comm.get(e["u"])
    v_comm = node_to_comm.get(e["v"])
    if u_comm in kept_set and v_comm in kept_set and u_comm != v_comm:
        w = float(e["weight"])
        if G.has_edge(u_comm, v_comm):
            G[u_comm][v_comm]["weight"] += w
        else:
            G.add_edge(u_comm, v_comm, weight=w)
        added += 1
print(f"Raw edges aggregated: {added:,}")
print(f"Graph pre-isolate-drop: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Optionally drop isolates (communities with degree 0 after aggregation)
if DROP_ISOLATES:
    isolates = [n for n, d in G.degree() if d == 0]
    if isolates:
        G.remove_nodes_from(isolates)
        print(f"Dropped {len(isolates)} isolates with degree 0: {isolates}")

print(f"Graph post-isolate-drop: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ===============================================
# Layout
# ===============================================
print("Computing layout...")
pos = nx.spring_layout(G, k=SPRING_K, iterations=SPRING_ITERS, seed=42)

# ===============================================
# Prepare node colors (viridis palette)
# ===============================================
nodes_list = list(G.nodes())
node_colors_arr = plt.cm.viridis(np.linspace(0, 1, len(nodes_list)))
node_color_map = {n: node_colors_arr[i] for i, n in enumerate(nodes_list)}

# ===============================================
# Edge drawing with true gradient between node colors
# ===============================================
fig, ax = plt.subplots(figsize=(18, 14), facecolor="white")
ax.set_facecolor("#f9f9f9")

if G.number_of_edges() > 0:
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    w_lo, w_hi = np.percentile(weights, EDGE_WEIGHT_PERC)
    if w_hi <= w_lo:
        w_lo, w_hi = weights.min(), weights.max()
    w_clip = np.clip(weights, w_lo, w_hi)
    w_norm = (w_clip - w_lo) / (w_hi - w_lo + 1e-12)
    line_widths_edge = EDGE_MIN_LW + (EDGE_MAX_LW - EDGE_MIN_LW) * w_norm
    alphas_edge = EDGE_ALPHA_MIN + (EDGE_ALPHA_MAX - EDGE_ALPHA_MIN) * w_norm

    # Build segments for all edges with gradient steps
    segs = []
    seg_colors = []
    seg_lws = []

    for (edge_idx, (u, v)) in enumerate(G.edges()):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        # interpolate coordinates
        ts = np.linspace(0.0, 1.0, EDGE_GRADIENT_STEPS)
        xs = x0 + (x1 - x0) * ts
        ys = y0 + (y1 - y0) * ts

        # node endpoint colors
        c1 = np.array(to_rgba(node_color_map[u]))
        c2 = np.array(to_rgba(node_color_map[v]))

        # alpha for this edge based on weight
        a = alphas_edge[edge_idx]
        c1[3] = a
        c2[3] = a

        # interpolate colors
        cs = np.linspace(c1, c2, EDGE_GRADIENT_STEPS)

        # create small segments with their own color
        for i in range(len(ts) - 1):
            segs.append([[xs[i], ys[i]], [xs[i+1], ys[i+1]]])
            seg_colors.append(cs[i])
            seg_lws.append(line_widths_edge[edge_idx])

    lc = LineCollection(segs, colors=seg_colors, linewidths=seg_lws,
                        capstyle="round", joinstyle="round", zorder=1, antialiased=True)
    ax.add_collection(lc)
else:
    print("No inter-community edges to draw.")

# ===============================================
# Node drawing
# ===============================================
sizes = np.array([G.nodes[n]["size"] for n in nodes_list], dtype=float)
node_sizes = np.sqrt(sizes) * NODE_SIZE_SCALE

for i, n in enumerate(nodes_list):
    x, y = pos[n]
    ax.scatter(x, y, s=node_sizes[i], c=[node_color_map[n]], alpha=0.92,
               edgecolors="black", linewidths=1.8, zorder=2)

    size_val = int(round(G.nodes[n]["size"]))
    fontsize = max(8, min(14, np.sqrt(size_val) / 30))
    ax.text(x, y, f"{size_val:,}", ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=3)

    ax.text(x, y - 0.08, f"C{n}", ha="center", va="top",
            fontsize=8, style="italic", color="#333", zorder=3)

# ===============================================
# Title / Legend / Framing
# ===============================================
total_nodes = int(sizes.sum())
num_edges = G.number_of_edges()
ax.set_title(
    f"Louvain Community Aggregation (> {MIN_COMMUNITY_SIZE} nodes, outliers removed {OUTLIER_SIZE_PERC[0]}-{OUTLIER_SIZE_PERC[1]}th)\n"
    f"{G.number_of_nodes()} communities | {total_nodes:,} nodes | {num_edges} inter-community connections",
    fontsize=18, fontweight="bold", pad=25
)
ax.axis("off")

# pad a bit
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
pad_x = (x1 - x0) * 0.06
pad_y = (y1 - y0) * 0.06
ax.set_xlim(x0 - pad_x, x1 + pad_x)
ax.set_ylim(y0 - pad_y, y1 + pad_y)

# legend (min/median/max sizes for stability)
size_examples = [sizes.min(), np.median(sizes), sizes.max()]
handles = []
labels = []
for s in size_examples:
    h = ax.scatter([], [], s=np.sqrt(s) * NODE_SIZE_SCALE, c="lightgray",
                   alpha=0.9, edgecolors="black", linewidths=1.5)
    handles.append(h)
    labels.append(f"{int(round(s)):,} nodes")
ax.legend(handles, labels, title="Community Size",
          loc="upper left", frameon=True, fontsize=11, title_fontsize=12)

plt.tight_layout()
output_file = "louvain_communities_viz.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {output_file}")
plt.show()

# ===============================================
# Summary stats
# ===============================================
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
trimmed_sorted = trimmed.sort_values("size", ascending=False)
print(f"Total communities (all):        {len(communities_df)}")
print(f"Communities > {MIN_COMMUNITY_SIZE}: {len(base)}")
print(f"Outliers removed outside {OUTLIER_SIZE_PERC[0]}-{OUTLIER_SIZE_PERC[1]}th: {len(dropped_outliers)}")
print(f"Communities shown after all filters: {len(G.nodes())}")
print(f"Total nodes visualized:         {total_nodes:,}")
print(f"Inter-community connections:    {num_edges}")
print(f"\nTop 5 largest communities shown:")
for _, row in trimmed_sorted.head(5).iterrows():
    print(f"  Community {row['community']:>4}: {int(round(row['size'])):>7,} nodes")
print(f"\nSmallest shown community:       {int(round(trimmed_sorted['size'].min())):,} nodes")
