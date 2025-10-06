
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.collections import LineCollection

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
# Keep communities strictly larger than 440 ("over 440")
large_communities = communities_df[communities_df['size'] > 440].copy()
print(f"Communities with >440 nodes: {len(large_communities)} / {len(communities_df)}")

if len(large_communities) == 0:
    print("No communities found with >440 nodes!")
    print(f"Max community size: {communities_df['size'].max()}")
    raise SystemExit(0)

# Map node -> community
node_to_comm = dict(zip(nodes_df['user_id'], nodes_df['community']))

# -----------------------------
# Build aggregated (community-level) graph
# -----------------------------
print("Building aggregated graph...")
G = nx.Graph()

# Add one node per large community with size attribute
for _, row in large_communities.iterrows():
    G.add_node(int(row['community']), size=int(row['size']))

# Precompute set for membership checks
large_comm_set = set(large_communities['community'].astype(int).tolist())

print("Computing inter-community connections...")
# Aggregate edge weights between communities
added = 0
skipped = 0
for _, e in edges_df.iterrows():
    u_comm = node_to_comm.get(e['u'])
    v_comm = node_to_comm.get(e['v'])

    # only connect different large communities
    if u_comm in large_comm_set and v_comm in large_comm_set and u_comm != v_comm:
        w = float(e['weight'])
        if G.has_edge(u_comm, v_comm):
            G[u_comm][v_comm]['weight'] += w
        else:
            G.add_edge(u_comm, v_comm, weight=w)
        added += 1
    else:
        skipped += 1
print(f"Edges aggregated: {added:,} (skipped raw edges: {skipped:,})")
print(f"Community nodes: {G.number_of_nodes()}, community edges: {G.number_of_edges()}")

# -----------------------------
# Visualization
# -----------------------------
print("Creating visualization...")
fig, ax = plt.subplots(figsize=(18, 14), facecolor='white')
ax.set_facecolor('#f9f9f9')

# Spring layout (slightly tighter spacing than before)
# If graph is small/large, moderate k by sqrt(n) to keep spacing balanced
n = max(G.number_of_nodes(), 1)
k_base = 1.2
k = k_base / np.sqrt(max(n, 1))
pos = nx.spring_layout(G, k=k, iterations=70, seed=42)

# -----------------------------
# Draw edges (normalized for visibility)
# -----------------------------
if G.number_of_edges() > 0:
    weights = np.array([G[u][v]['weight'] for u, v in G.edges()], dtype=float)
    w_min, w_max = weights.min(), weights.max()
    # Normalize to [0,1] robustly
    denom = (w_max - w_min) if (w_max > w_min) else 1.0
    norm = (weights - w_min) / denom

    # Thickness and alpha scale with normalized weight
    line_widths = 0.6 + 3.2 * norm        # 0.6..3.8
    alphas = 0.25 + 0.65 * norm           # 0.25..0.90

    # Color map: use matplotlib Blues; compose RGBA with per-edge alpha
    cmap = plt.cm.Blues
    colors = cmap(norm)
    colors[:, 3] = alphas  # set alpha channel

    # Build list of line segments
    segs = []
    for (u, v) in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        segs.append([(x0, y0), (x1, y1)])

    lc = LineCollection(segs, colors=colors, linewidths=line_widths, capstyle='round', zorder=1)
    ax.add_collection(lc)
else:
    print("No inter-community edges to draw.")

# -----------------------------
# Draw nodes
# -----------------------------
# Node sizes (sqrt scaling)
sizes = np.array([G.nodes[n]['size'] for n in G.nodes()], dtype=float)
node_sizes = np.sqrt(sizes) * 15.0

# Node colors: distinct-ish via viridis sampling
node_colors = plt.cm.viridis(np.linspace(0, 1, G.number_of_nodes()))

for i, (node, (x, y)) in enumerate(pos.items()):
    ax.scatter(x, y, s=node_sizes[i], c=[node_colors[i]],
               alpha=0.9, edgecolors='black', linewidths=1.8, zorder=2)

    # Size label inside the node
    size_val = int(G.nodes[node]['size'])
    fontsize = max(8, min(14, np.sqrt(size_val) / 30))
    ax.text(x, y, f"{size_val:,}", ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white', zorder=3)

    # Community ID below node
    ax.text(x, y - 0.08, f"C{node}", ha='center', va='top',
            fontsize=8, style='italic', color='#333', zorder=3)

# -----------------------------
# Title / Stats / Legend
# -----------------------------
total_nodes = int(sizes.sum())
num_edges = G.number_of_edges()

ax.set_title(
    f"Louvain Community Aggregation (>440 nodes)\n"
    f"{G.number_of_nodes()} communities | {total_nodes:,} nodes | {num_edges} inter-community connections",
    fontsize=18, fontweight='bold', pad=25
)
ax.axis('off')

# Pad view a bit
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
pad_x = (x1 - x0) * 0.05
pad_y = (y1 - y0) * 0.05
ax.set_xlim(x0 - pad_x, x1 + pad_x)
ax.set_ylim(y0 - pad_y, y1 + pad_y)

# Legend for node sizes (use quantiles for representative examples)
quantiles = np.clip(np.quantile(sizes, [0.1, 0.5, 0.9]), sizes.min(), sizes.max())
legend_handles = []
legend_labels = []
for s in quantiles:
    handle = ax.scatter([], [], s=np.sqrt(s) * 15.0, c='lightgray',
                        alpha=0.9, edgecolors='black', linewidths=1.5)
    legend_handles.append(handle)
    legend_labels.append(f"{int(s):,} nodes")
ax.legend(legend_handles, legend_labels, title='Community Size',
          loc='upper left', frameon=True, fontsize=11, title_fontsize=12)

plt.tight_layout()
output_file = 'louvain_communities_viz.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_file}")
plt.show()

# -----------------------------
# Summary statistics
# -----------------------------
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
large_sorted = large_communities.sort_values('size', ascending=False)
print(f"Total communities (all):        {len(communities_df)}")
print(f"Communities shown (>440):       {len(large_communities)}")
print(f"Total nodes visualized:         {total_nodes:,}")
print(f"Inter-community connections:    {num_edges}")
print(f"\nTop 5 largest communities:")
for _, row in large_sorted.head(5).iterrows():
    print(f"  Community {int(row['community']):>4}: {int(row['size']):>7,} nodes")
print(f"\nSmallest shown community:       {int(large_sorted['size'].min()):,} nodes")
