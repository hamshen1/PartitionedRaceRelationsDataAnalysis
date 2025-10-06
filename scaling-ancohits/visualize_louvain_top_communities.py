
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

# -----------------------------
# Config (tweak here if needed)
# -----------------------------
MIN_COMMUNITY_SIZE = 440          # keep communities with size > this
MAX_COMMUNITIES = None            # e.g., 25 to keep the largest 25 communities after filtering
SPRING_K = 3.0                    # wide spacing to avoid overlap (restores your original intent)
SPRING_ITERS = 100
NODE_SIZE_SCALE = 15.0            # sqrt(size) * scale (kept consistent with original)
EDGE_MIN_LW = 1.2                 # minimum line width (in points)
EDGE_MAX_LW = 5.0                 # maximum line width
EDGE_ALPHA_MIN = 0.35             # min alpha for faint edges
EDGE_ALPHA_MAX = 0.95             # max alpha for strong edges
EDGE_LO = 5                       # robust low percentile for weight normalization
EDGE_HI = 95                      # robust high percentile for weight normalization
EDGE_CMAP = plt.cm.Blues          # color map for edges

# -----------------------------
# Load data
# -----------------------------
print("Loading data...")
communities_df = pd.read_csv("./louvain_out/communities.csv")
nodes_df = pd.read_csv("./louvain_out/nodes.csv")
edges_df = pd.read_csv("./louvain_out/edges.csv")

# -----------------------------
# Filter communities
# -----------------------------
large = communities_df[communities_df["size"] > MIN_COMMUNITY_SIZE].copy()
if MAX_COMMUNITIES is not None and MAX_COMMUNITIES > 0 and len(large) > MAX_COMMUNITIES:
    large = large.sort_values("size", ascending=False).head(MAX_COMMUNITIES).copy()

print(f"Communities kept (> {MIN_COMMUNITY_SIZE}): {len(large)} / {len(communities_df)}")
if len(large) == 0:
    print("No communities meet the criterion.")
    print(f"Max community size in data: {communities_df['size'].max()}")
    raise SystemExit(0)

# node -> community map
node_to_comm = dict(zip(nodes_df["user_id"], nodes_df["community"]))

# -----------------------------
# Build aggregated graph
# -----------------------------
print("Building aggregated graph...")
G = nx.Graph()
for _, row in large.iterrows():
    # preserve sizes as-is; they should already be numeric
    G.add_node(row["community"], size=row["size"])

large_set = set(large["community"].tolist())

# aggregate inter-community weights
added = 0
for _, e in edges_df.iterrows():
    u_comm = node_to_comm.get(e["u"])
    v_comm = node_to_comm.get(e["v"])
    if u_comm in large_set and v_comm in large_set and u_comm != v_comm:
        w = float(e["weight"])
        if G.has_edge(u_comm, v_comm):
            G[u_comm][v_comm]["weight"] += w
        else:
            G.add_edge(u_comm, v_comm, weight=w)
        added += 1
print(f"Raw edges aggregated: {added:,}")
print(f"Community graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# -----------------------------
# Layout
# -----------------------------
print("Computing layout...")
pos = nx.spring_layout(G, k=SPRING_K, iterations=SPRING_ITERS, seed=42)  # restore wide spacing

# -----------------------------
# Edge drawing (robust normalization)
# -----------------------------
fig, ax = plt.subplots(figsize=(18, 14), facecolor="white")
ax.set_facecolor("#f9f9f9")

# Edge drawing with gradient colors between node colors
if G.number_of_edges() > 0:
    node_color_map = dict(zip(G.nodes(), plt.cm.viridis(np.linspace(0, 1, G.number_of_nodes()))))
    
    # collect all segments
    segs = []
    seg_colors = []
    
    weights = np.array([G[u][v]['weight'] for u, v in G.edges()], dtype=float)
    lo, hi = np.percentile(weights, [5, 95])
    w_clipped = np.clip(weights, lo, hi)
    norm = (w_clipped - lo) / (hi - lo + 1e-12)
    lws = EDGE_MIN_LW + (EDGE_MAX_LW - EDGE_MIN_LW) * norm
    
    for (u, v), lw in zip(G.edges(), lws):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        segs.append([[x0, y0], [x1, y1]])

        # interpolate node colors 0→1
        c1 = np.array(to_rgba(node_color_map[u]))
        c2 = np.array(to_rgba(node_color_map[v]))
        grad = np.linspace(c1, c2, 2)
        seg_colors.append(grad)

    # Build a LineCollection with gradient edges
    lc = LineCollection(segs, linewidths=lws, zorder=1, antialiased=True)
    lc.set_array(None)  # disable colormap scaling
    lc.set_color([np.mean(sc, axis=0) for sc in seg_colors])  # average color for each edge (simpler version)
    ax.add_collection(lc)
    
# -----------------------------
# Node drawing
# -----------------------------
sizes = np.array([G.nodes[n]["size"] for n in G.nodes()], dtype=float)
node_sizes = np.sqrt(sizes) * NODE_SIZE_SCALE  # same functional form

node_colors = plt.cm.viridis(np.linspace(0, 1, G.number_of_nodes()))

for i, (node, (x, y)) in enumerate(pos.items()):
    ax.scatter(x, y, s=node_sizes[i], c=[node_colors[i]], alpha=0.92,
               edgecolors="black", linewidths=1.8, zorder=2)

    size_val = int(round(G.nodes[node]["size"]))
    fontsize = max(8, min(14, np.sqrt(size_val) / 30))
    ax.text(x, y, f"{size_val:,}", ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=3)

    ax.text(x, y - 0.08, f"C{node}", ha="center", va="top",
            fontsize=8, style="italic", color="#333", zorder=3)

# -----------------------------
# Title / Legend / Framing
# -----------------------------
total_nodes = int(sizes.sum())
num_edges = G.number_of_edges()
ax.set_title(
    f"Louvain Community Aggregation (> {MIN_COMMUNITY_SIZE} nodes)\n"
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

# -----------------------------
# Summary stats
# -----------------------------
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
large_sorted = large.sort_values("size", ascending=False)
print(f"Total communities (all):        {len(communities_df)}")
print(f"Communities shown (> {MIN_COMMUNITY_SIZE}): {len(large)}")
print(f"Total nodes visualized:         {total_nodes:,}")
print(f"Inter-community connections:    {num_edges}")
print(f"\nTop 5 largest communities:")
for _, row in large_sorted.head(5).iterrows():
    print(f"  Community {row['community']:>4}: {int(round(row['size'])):>7,} nodes")
print(f"\nSmallest shown community:       {int(round(large_sorted['size'].min())):,} nodes")
