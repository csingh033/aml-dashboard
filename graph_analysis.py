import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def build_transaction_graph(df):
    """
    Build a directed graph from transaction data.

    Args:
        df: DataFrame with transaction data containing columns:
            - customer_no_hashed: sender customer ID
            - beneficiary_name_hashed: receiver customer ID
            - amount: transaction amount
            - transfer_type: type of transfer
            - createdDateTime: transaction timestamp
            - reference_no: transaction reference

    Returns:
        networkx.DiGraph: Directed graph representing transaction network
    """
    G = nx.DiGraph()

    for _, row in df.iterrows():
        sender = row["customer_no_hashed"]
        receiver = row["beneficiary_name_hashed"]

        # Skip if either sender or receiver is missing
        if pd.isna(sender) or pd.isna(receiver) or sender == "" or receiver == "":
            continue

        # Add edge with transaction details as attributes
        G.add_edge(
            sender,
            receiver,
            amount=row["amount"],
            transfer_type=row["transfer_type"],
            created=row["createdDateTime"],
            reference_no=row["reference_no"],
        )

    return G


def detect_suspicious_hubs(G, min_degree=5):
    """
    Detect suspicious hub nodes based on their degree (number of connections).

    Args:
        G: networkx.DiGraph - Transaction graph
        min_degree: int - Minimum degree threshold for suspicious hubs

    Returns:
        list: List of node IDs that are suspicious hubs
    """
    # Get nodes with high degree (many connections)
    high_degree_nodes = [n for n, d in G.degree() if d >= min_degree]

    # Also check for nodes with high in-degree (receiving many transactions)
    high_in_degree_nodes = [n for n, d in G.in_degree() if d >= min_degree]

    # Combine and remove duplicates
    suspicious_hubs = list(set(high_degree_nodes + high_in_degree_nodes))

    return suspicious_hubs


def find_cycles(G):
    """
    Find cycles in the transaction graph.
    Cycles can indicate potential money laundering patterns.

    Args:
        G: networkx.DiGraph - Transaction graph

    Returns:
        list: List of cycles, where each cycle is a list of node IDs
    """
    try:
        # Find all simple cycles in the graph
        cycles = list(nx.simple_cycles(G))
        return cycles
    except Exception as e:
        # If there's an error (e.g., graph is too large), return empty list
        print(f"Error finding cycles: {e}")
        return []


def visualize_graph(
    G,
    pos=None,
    node_size=400,
    node_color="lightblue",
    edge_color="gray",
    figsize=(10, 8),
    title="Transaction Network",
):
    """
    Visualize the transaction graph using matplotlib.

    Args:
        G: networkx.DiGraph - Transaction graph
        pos: dict - Node positions (if None, will use spring layout)
        node_size: int - Size of nodes in the visualization
        node_color: str - Color of nodes
        edge_color: str - Color of edges
        figsize: tuple - Figure size (width, height)
        title: str - Title for the plot

    Returns:
        tuple: (fig, ax) - matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use spring layout if no positions provided
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=False,  # We'll add labels separately for better control
        node_size=node_size,
        node_color=node_color,
        edge_color=edge_color,
        ax=ax,
        arrows=True,  # Show direction of edges
        arrowsize=10,
    )

    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # Add title
    ax.set_title(title)

    return fig, ax


def analyze_graph_metrics(G):
    """
    Analyze various metrics of the transaction graph.

    Args:
        G: networkx.DiGraph - Transaction graph

    Returns:
        dict: Dictionary containing various graph metrics
    """
    if len(G.nodes()) == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0,
            "avg_degree": 0,
            "max_degree": 0,
            "num_cycles": 0,
            "suspicious_hubs": [],
        }

    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / len(G.nodes()),
        "max_degree": max(dict(G.degree()).values()),
        "num_cycles": len(find_cycles(G)),
        "suspicious_hubs": detect_suspicious_hubs(G),
    }

    return metrics
