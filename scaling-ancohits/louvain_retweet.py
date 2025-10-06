#!/usr/bin/env python3
"""
Louvain community detection on Twitter retweet interactions.

Edges are formed from retweeter -> original author for rows where tweet_type == "retweet".
Edges are aggregated and weighted by the number of retweets between a pair.

Usage:
  pip install pandas networkx python-louvain
  python louvain_retweet.py --input /path/to/data.csv --outdir ./out --min-weight 2
"""

import argparse
import os
from collections import Counter

import pandas as pd
import networkx as nx

import sys
try:
    import community as community_louvain  # python-louvain
except Exception:
    print("ERROR: Install dependency: pip install python-louvain", file=sys.stderr)
    raise

def parse_args():
    p = argparse.ArgumentParser(description="Louvain communities from retweet interactions")
    p.add_argument("--input", required=True, help="Path to CSV of tweet interactions")
    p.add_argument("--outdir", default="./louvain_out", help="Directory to write outputs")
    # Column names (defaults match the provided example CSV)
    p.add_argument("--tweet-type-col", default="tweet_type")
    p.add_argument("--user-id-col", default="user_id")
    p.add_argument("--username-col", default="username")
    p.add_argument("--parent-user-id-col", default="parent_user_id")
    p.add_argument("--parent-username-col", default="parent_username")
    # Retweet label in the tweet_type column
    p.add_argument("--retweet-label", default="retweet")
    # Graph/algorithm parameters
    p.add_argument("--directed", action="store_true", help="Keep directed edges in exports (Louvain still uses undirected projection)")
    p.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution parameter")
    p.add_argument("--min-weight", type=float, default=1.0, help="Drop edges with weight < this threshold after aggregation")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for Louvain")
    return p.parse_args()

def load_and_clean(path, cols):
    # Read with default dtype inference (no dtype forcing)
    df = pd.read_csv(path)
    # Unconditionally drop the first *data* row (not the header),
    # because the export included a label row as the first record.
    if len(df) > 0:
        df = df.iloc[1:].reset_index(drop=True)

    # Validate required columns
    for key, col in cols.items():
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in input CSV. Available columns: {list(df.columns)}")

    return df

def aggregate_edges(df, cols, retweet_label, min_weight):
    # Filter to retweets (case-insensitive, but keeps dtype inference otherwise)
    tt = df[cols['tweet_type']].astype(str).str.strip().str.lower()
    df_rt = df[tt == str(retweet_label).strip().lower()].copy()
    if df_rt.empty:
        raise ValueError(f"No retweet rows found with label {retweet_label!r} in column {cols['tweet_type']!r}")

    # Build retweeter -> original author pairs
    u = df_rt[cols['user_id']].astype(str).str.strip()
    v = df_rt[cols['parent_user_id']].astype(str).str.strip()

    pairs = pd.DataFrame({"u": u, "v": v})
    # Count occurrences to make weights
    edges = pairs.value_counts().reset_index(name="weight")
    # Remove self-loops
    edges = edges[edges["u"] != edges["v"]]
    # Apply minimum weight threshold
    edges = edges[edges["weight"] >= min_weight].reset_index(drop=True)
    return edges, df_rt

def make_graph(edges, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    for _, row in edges.iterrows():
        u, v, w = row["u"], row["v"], float(row["weight"])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G

def project_to_undirected(G):
    if not G.is_directed():
        return G
    UG = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if UG.has_edge(u, v):
            UG[u][v]["weight"] += w
        else:
            UG.add_edge(u, v, weight=w)
    return UG

def attach_node_attrs(G, df_rt, cols):
    id_to_username = {}
    if cols['user_id'] in df_rt.columns and cols['username'] in df_rt.columns:
        m1 = df_rt[[cols['user_id'], cols['username']]].dropna().astype(str).drop_duplicates()
        id_to_username.update(dict(zip(m1[cols['user_id']], m1[cols['username']])))
    if cols['parent_user_id'] in df_rt.columns and cols['parent_username'] in df_rt.columns:
        m2 = df_rt[[cols['parent_user_id'], cols['parent_username']]].dropna().astype(str).drop_duplicates()
        id_to_username.update(dict(zip(m2[cols['parent_user_id']], m2[cols['parent_username']])))

    nx.set_node_attributes(G, {n: {"username": id_to_username.get(str(n), None)} for n in G.nodes()})
    deg = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    nx.set_node_attributes(G, deg, "degree")
    nx.set_node_attributes(G, strength, "strength")

def run_louvain(UG, resolution=1.0, random_state=42):
    partition = community_louvain.best_partition(UG, weight="weight", resolution=resolution, random_state=random_state)
    Q = community_louvain.modularity(partition, UG, weight="weight")
    return partition, Q

def export_outputs(outdir, edges, G, UG, partition, modularity):
    os.makedirs(outdir, exist_ok=True)
    edges.to_csv(os.path.join(outdir, "edges.csv"), index=False)

    rows = []
    for n in G.nodes():
        data = G.nodes[n]
        rows.append({
            "user_id": n,
            "username": data.get("username"),
            "degree": data.get("degree"),
            "strength": data.get("strength"),
            "community": partition.get(n)
        })
    nodes_df = pd.DataFrame(rows).sort_values(["community", "strength"], ascending=[True, False])
    nodes_df.to_csv(os.path.join(outdir, "nodes.csv"), index=False)

    counts = Counter(partition.values())
    pd.DataFrame([{"community": c, "size": s} for c, s in sorted(counts.items(), key=lambda x: (-x[1], x[0]))])\
      .to_csv(os.path.join(outdir, "communities.csv"), index=False)

    nx.write_gexf(UG, os.path.join(outdir, "graph.gexf"))
    with open(os.path.join(outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Nodes: {UG.number_of_nodes()}\n")
        f.write(f"Edges: {UG.number_of_edges()}\n")
        f.write(f"Modularity (Louvain): {modularity:.6f}\n")

def main():
    args = parse_args()
    cols = {
        "tweet_type": args.tweet_type_col,
        "user_id": args.user_id_col,
        "username": args.username_col,
        "parent_user_id": args.parent_user_id_col,
        "parent_username": args.parent_username_col,
    }

    df = load_and_clean(args.input, cols)
    edges, df_rt = aggregate_edges(df, cols, args.retweet_label, args.min_weight)

    G = make_graph(edges, directed=args.directed)
    UG = project_to_undirected(G)

    attach_node_attrs(G, df_rt, cols)
    partition, Q = run_louvain(UG, resolution=args.resolution, random_state=args.random_state)

    export_outputs(args.outdir, edges, G, UG, partition, Q)

    print(f"Done. Outputs written to: {args.outdir}")
    print("Files: edges.csv, nodes.csv, communities.csv, graph.gexf, metrics.txt")

if __name__ == "__main__":
    main()
