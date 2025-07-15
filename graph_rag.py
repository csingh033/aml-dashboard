import networkx as nx
import pandas as pd


def build_graph_from_df(df):
    """
    Build a directed graph from DataFrame for RAG analysis.

    Args:
        df: DataFrame with transaction data

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


def extract_graph_context(G, target_node=None):
    """
    Extract context from the graph for RAG analysis.

    Args:
        G: networkx.DiGraph - Transaction graph
        target_node: str - Specific node to focus on (optional)

    Returns:
        dict: Dictionary containing graph context information
    """
    context = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_list": list(G.nodes()),
        "edge_list": list(G.edges(data=True)),
    }

    if target_node and target_node in G.nodes():
        # Extract context specific to target node
        neighbors = list(G.neighbors(target_node))
        predecessors = list(G.predecessors(target_node))

        context["target_node"] = target_node
        context["neighbors"] = neighbors
        context["predecessors"] = predecessors
        context["in_degree"] = G.in_degree(target_node)
        context["out_degree"] = G.out_degree(target_node)

        # Get edge data for target node
        target_edges = []
        for u, v, data in G.edges(data=True):
            if u == target_node or v == target_node:
                target_edges.append(
                    {
                        "from": u,
                        "to": v,
                        "amount": data.get("amount", 0),
                        "transfer_type": data.get("transfer_type", ""),
                        "created": data.get("created", ""),
                        "reference_no": data.get("reference_no", ""),
                    }
                )
        context["target_edges"] = target_edges

    return context


def format_rag_prompt(customer_id, transaction_data, graph_context=None):
    """
    Format a RAG prompt for LLM analysis of a customer's transactions.

    Args:
        customer_id: str - Hashed customer ID
        transaction_data: list - List of transaction dictionaries
        graph_context: dict - Graph context information (optional)

    Returns:
        str: Formatted prompt for LLM analysis
    """
    prompt = f"""
**AML Analysis Request for Customer: {customer_id}**

**Transaction Data:**
{transaction_data}

"""

    if graph_context:
        prompt += f"""
**Graph Context:**
- Total nodes in network: {graph_context.get('total_nodes', 0)}
- Total edges in network: {graph_context.get('total_edges', 0)}
- Customer connections: {len(graph_context.get('neighbors', []))} outgoing, {len(graph_context.get('predecessors', []))} incoming
- Customer degree: {graph_context.get('in_degree', 0)} incoming, {graph_context.get('out_degree', 0)} outgoing

"""

    prompt += """
**Analysis Request:**
Please provide a concise AML analysis for this customer with:

1. **Risk Assessment:** 
   - Overall risk level: HIGH/MEDIUM/LOW
   - Use RED for HIGH risk, GREEN for LOW risk, BLUE for MEDIUM/NEUTRAL

2. **4 Key Bullet Points Only:**
   - Most critical findings
   - Key risk indicators
   - Suspicious patterns
   - Recommended actions

Keep the analysis concise and actionable. Focus on the most important AML concerns.
"""

    return prompt
