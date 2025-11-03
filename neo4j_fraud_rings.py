"""
Neo4j Fraud Ring Detection Module

This module provides functionality for:
- Connecting to Neo4j database
- Building fraud ring graphs using consolidated beneficiary data
- Detecting suspicious transaction patterns and rings
- Providing explainable fraud risk analysis
"""

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict


class Neo4jFraudDetector:
    """Neo4j-powered fraud ring detection system"""

    def __init__(
        self,
        uri: str = "neo4j://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.connected = False

    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {str(e)}")
            self.connected = False
            return False

    def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.connected = False

    def clear_database(self):
        """Clear all nodes and relationships (for fresh analysis)"""
        if not self.connected:
            return False

        with self.driver.session() as session:
            try:
                session.run("MATCH (n) DETACH DELETE n")
                return True
            except Exception as e:
                st.error(f"Failed to clear database: {str(e)}")
                return False

    def create_constraints_and_indexes(self):
        """Create constraints and indexes for better performance"""
        if not self.connected:
            return False

        constraints_and_indexes = [
            "CREATE CONSTRAINT sender_id IF NOT EXISTS FOR (s:Sender) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT beneficiary_id IF NOT EXISTS FOR (b:Beneficiary) REQUIRE b.id IS UNIQUE",
            "CREATE INDEX sender_name_idx IF NOT EXISTS FOR (s:Sender) ON (s.name)",
            "CREATE INDEX beneficiary_name_idx IF NOT EXISTS FOR (b:Beneficiary) ON (b.name)",
            "CREATE INDEX transaction_amount_idx IF NOT EXISTS FOR ()-[t:SENT_TO]-() ON (t.amount)",
        ]

        with self.driver.session() as session:
            for constraint in constraints_and_indexes:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    pass
        return True

    def build_fraud_graph(
        self, df: pd.DataFrame, beneficiary_consolidation_map: Dict[str, str] = None
    ):
        """
        Build fraud detection graph in Neo4j using transaction data and consolidated beneficiaries

        Args:
            df: Transaction DataFrame
            beneficiary_consolidation_map: Mapping from original to consolidated beneficiary names
        """
        if not self.connected:
            st.error("Not connected to Neo4j database")
            return False

        # Apply beneficiary consolidation if provided
        df_consolidated = df.copy()
        if beneficiary_consolidation_map:
            df_consolidated["beneficiary_name_consolidated"] = (
                df_consolidated["beneficiary_name"]
                .map(beneficiary_consolidation_map)
                .fillna(df_consolidated["beneficiary_name"])
            )
        else:
            df_consolidated["beneficiary_name_consolidated"] = df_consolidated[
                "beneficiary_name"
            ]

        # Create progress tracker
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            with self.driver.session() as session:
                # Step 1: Create sender nodes
                status_text.text("Creating sender nodes...")
                senders = (
                    df_consolidated.groupby("customer_no")
                    .agg(
                        {
                            "CustomerName": "first",
                            "amount": ["sum", "count", "mean"],
                            "beneficiary_name_consolidated": "nunique",
                            "transfer_type": lambda x: list(x.unique()),
                        }
                    )
                    .reset_index()
                )

                # Flatten column names
                senders.columns = [
                    "customer_no",
                    "customer_name",
                    "total_sent",
                    "transaction_count",
                    "avg_amount",
                    "unique_beneficiaries",
                    "transfer_types",
                ]

                for idx, sender in senders.iterrows():
                    session.run(
                        """
                        MERGE (s:Sender {id: $customer_no})
                        SET s.name = $customer_name,
                            s.total_sent = $total_sent,
                            s.transaction_count = $transaction_count,
                            s.avg_amount = $avg_amount,
                            s.unique_beneficiaries = $unique_beneficiaries,
                            s.transfer_types = $transfer_types,
                            s.risk_score = $risk_score
                    """,
                        {
                            "customer_no": sender["customer_no"],
                            "customer_name": sender["customer_name"] or "Unknown",
                            "total_sent": float(sender["total_sent"]),
                            "transaction_count": int(sender["transaction_count"]),
                            "avg_amount": float(sender["avg_amount"]),
                            "unique_beneficiaries": int(sender["unique_beneficiaries"]),
                            "transfer_types": sender["transfer_types"],
                            "risk_score": self._calculate_sender_risk_score(sender),
                        },
                    )

                progress_bar.progress(0.3)

                # Step 2: Create beneficiary nodes (using consolidated names)
                status_text.text("Creating beneficiary nodes...")
                beneficiaries = (
                    df_consolidated.groupby("beneficiary_name_consolidated")
                    .agg(
                        {
                            "amount": ["sum", "count", "mean"],
                            "customer_no": "nunique",
                            "transfer_type": lambda x: list(x.unique()),
                            "beneficiary_name": lambda x: list(
                                x.unique()
                            ),  # Original names
                        }
                    )
                    .reset_index()
                )

                # Flatten column names
                beneficiaries.columns = [
                    "beneficiary_name_consolidated",
                    "total_received",
                    "transaction_count",
                    "avg_amount",
                    "unique_senders",
                    "transfer_types",
                    "original_names",
                ]

                for idx, beneficiary in beneficiaries.iterrows():
                    session.run(
                        """
                        MERGE (b:Beneficiary {id: $beneficiary_name})
                        SET b.name = $beneficiary_name,
                            b.original_names = $original_names,
                            b.total_received = $total_received,
                            b.transaction_count = $transaction_count,
                            b.avg_amount = $avg_amount,
                            b.unique_senders = $unique_senders,
                            b.transfer_types = $transfer_types,
                            b.risk_score = $risk_score,
                            b.is_consolidated = $is_consolidated
                    """,
                        {
                            "beneficiary_name": beneficiary[
                                "beneficiary_name_consolidated"
                            ],
                            "original_names": beneficiary["original_names"],
                            "total_received": float(beneficiary["total_received"]),
                            "transaction_count": int(beneficiary["transaction_count"]),
                            "avg_amount": float(beneficiary["avg_amount"]),
                            "unique_senders": int(beneficiary["unique_senders"]),
                            "transfer_types": beneficiary["transfer_types"],
                            "risk_score": self._calculate_beneficiary_risk_score(
                                beneficiary
                            ),
                            "is_consolidated": len(beneficiary["original_names"]) > 1,
                        },
                    )

                progress_bar.progress(0.6)

                # Step 3: Create relationships
                status_text.text("Creating transaction relationships...")
                transaction_count = 0
                total_transactions = len(df_consolidated)

                for idx, transaction in df_consolidated.iterrows():
                    session.run(
                        """
                        MATCH (s:Sender {id: $sender_id})
                        MATCH (b:Beneficiary {id: $beneficiary_id})
                        MERGE (s)-[r:SENT_TO]->(b)
                        ON CREATE SET r.first_transaction = $datetime,
                                     r.transaction_count = 1,
                                     r.total_amount = $amount,
                                     r.transfer_types = [$transfer_type]
                        ON MATCH SET r.last_transaction = $datetime,
                                    r.transaction_count = r.transaction_count + 1,
                                    r.total_amount = r.total_amount + $amount,
                                    r.transfer_types = CASE 
                                        WHEN $transfer_type IN r.transfer_types 
                                        THEN r.transfer_types
                                        ELSE r.transfer_types + [$transfer_type]
                                    END
                    """,
                        {
                            "sender_id": transaction["customer_no"],
                            "beneficiary_id": transaction[
                                "beneficiary_name_consolidated"
                            ],
                            "amount": float(transaction["amount"]),
                            "datetime": transaction["createdDateTime"].isoformat(),
                            "transfer_type": transaction["transfer_type"],
                        },
                    )

                    transaction_count += 1
                    if transaction_count % 100 == 0:
                        progress_bar.progress(
                            0.6 + (transaction_count / total_transactions) * 0.3
                        )

                progress_bar.progress(0.9)

                # Step 4: Calculate relationship risk scores
                status_text.text("Calculating relationship risk scores...")
                session.run(
                    """
                    MATCH (s:Sender)-[r:SENT_TO]->(b:Beneficiary)
                    SET r.risk_score = 
                        CASE 
                            WHEN r.total_amount > 50000 THEN 0.8
                            WHEN r.transaction_count > 10 THEN 0.7
                            WHEN b.unique_senders > 5 THEN 0.6
                            ELSE 0.3
                        END + 
                        CASE 
                            WHEN 'INTERNATIONAL_PAYMENT' IN r.transfer_types THEN 0.2
                            ELSE 0.0
                        END
                """
                )

                progress_bar.progress(1.0)
                status_text.text("✅ Graph creation completed!")

                return True

        except Exception as e:
            st.error(f"Error building fraud graph: {str(e)}")
            return False

    def _calculate_sender_risk_score(self, sender_data):
        """Calculate risk score for sender based on transaction patterns"""
        score = 0.0

        # High total amount sent
        if sender_data["total_sent"] > 100000:
            score += 0.3
        elif sender_data["total_sent"] > 50000:
            score += 0.2

        # Many unique beneficiaries (potential money mule)
        if sender_data["unique_beneficiaries"] > 10:
            score += 0.3
        elif sender_data["unique_beneficiaries"] > 5:
            score += 0.2

        # High transaction frequency
        if sender_data["transaction_count"] > 50:
            score += 0.2
        elif sender_data["transaction_count"] > 20:
            score += 0.1

        # International payments
        if "INTERNATIONAL_PAYMENT" in sender_data["transfer_types"]:
            score += 0.2

        return min(score, 1.0)

    def _calculate_beneficiary_risk_score(self, beneficiary_data):
        """Calculate risk score for beneficiary based on receiving patterns"""
        score = 0.0

        # Multiple senders (potential money mule)
        if beneficiary_data["unique_senders"] > 10:
            score += 0.4
        elif beneficiary_data["unique_senders"] > 5:
            score += 0.3
        elif beneficiary_data["unique_senders"] > 3:
            score += 0.2

        # High total received
        if beneficiary_data["total_received"] > 100000:
            score += 0.3
        elif beneficiary_data["total_received"] > 50000:
            score += 0.2

        # Name consolidation (multiple name variations)
        if len(beneficiary_data["original_names"]) > 1:
            score += 0.3

        return min(score, 1.0)

    def detect_fraud_rings(self, min_ring_size: int = 3, max_rings: int = 10):
        """
        Detect potential fraud rings using Neo4j graph algorithms

        Args:
            min_ring_size: Minimum number of nodes in a ring
            max_rings: Maximum number of rings to return
        """
        if not self.connected:
            return []

        fraud_rings = []

        with self.driver.session() as session:
            # Pattern 1: Circular money flows (A -> B -> C -> A)
            cycles = session.run(
                """
                MATCH path = (s1:Sender)-[:SENT_TO]->(b1:Beneficiary)<-[:SENT_TO]-(s2:Sender)-[:SENT_TO]->(b2:Beneficiary)<-[:SENT_TO]-(s1)
                WHERE s1 <> s2 AND b1 <> b2
                AND length(path) >= $min_size
                WITH path, 
                     [node in nodes(path) | node.id] as ring_members,
                     reduce(total = 0, rel in relationships(path) | total + rel.total_amount) as total_amount,
                     reduce(risk_total = 0, rel in relationships(path) | risk_total + rel.risk_score) as total_risk_score
                RETURN path, ring_members, total_amount, total_risk_score
                ORDER BY total_risk_score DESC
                LIMIT $max_rings
            """,
                min_size=min_ring_size * 2,
                max_rings=max_rings,
            ).data()

            for cycle in cycles:
                fraud_rings.append(
                    {
                        "type": "Circular Flow",
                        "members": cycle["ring_members"],
                        "total_amount": cycle["total_amount"],
                        "risk_score": cycle["total_risk_score"],
                        "pattern": "Money flows in circular pattern between entities",
                    }
                )

            # Pattern 2: Hub and spoke (one beneficiary receiving from many senders)
            hubs = session.run(
                """
                MATCH (b:Beneficiary)<-[r:SENT_TO]-(s:Sender)
                WITH b, count(s) as sender_count, sum(r.total_amount) as total_received,
                     collect({sender: s.id, amount: r.total_amount, risk: r.risk_score}) as connections
                WHERE sender_count >= $min_senders AND b.risk_score > 0.5
                RETURN b.id as hub_beneficiary, b.name as hub_name, sender_count, total_received,
                       connections, b.risk_score as hub_risk_score
                ORDER BY hub_risk_score DESC
                LIMIT $max_rings
            """,
                min_senders=min_ring_size,
                max_rings=max_rings,
            ).data()

            for hub in hubs:
                fraud_rings.append(
                    {
                        "type": "Hub Pattern",
                        "hub": hub["hub_beneficiary"],
                        "hub_name": hub["hub_name"],
                        "sender_count": hub["sender_count"],
                        "total_amount": hub["total_received"],
                        "connections": hub["connections"],
                        "risk_score": hub["hub_risk_score"],
                        "pattern": f"Single beneficiary receiving from {hub['sender_count']} different senders",
                    }
                )

            # Pattern 3: Rapid fire transactions (same sender to same beneficiary, high frequency)
            rapid_fire = session.run(
                """
                MATCH (s:Sender)-[r:SENT_TO]->(b:Beneficiary)
                WHERE r.transaction_count > 10 AND r.risk_score > 0.6
                RETURN s.id as sender, s.name as sender_name,
                       b.id as beneficiary, b.name as beneficiary_name,
                       r.transaction_count, r.total_amount, r.risk_score
                ORDER BY r.risk_score DESC
                LIMIT $max_rings
            """,
                max_rings=max_rings,
            ).data()

            for rf in rapid_fire:
                fraud_rings.append(
                    {
                        "type": "Rapid Fire",
                        "sender": rf["sender"],
                        "sender_name": rf["sender_name"],
                        "beneficiary": rf["beneficiary"],
                        "beneficiary_name": rf["beneficiary_name"],
                        "transaction_count": rf["transaction_count"],
                        "total_amount": rf["total_amount"],
                        "risk_score": rf["risk_score"],
                        "pattern": f"High frequency transactions ({rf['transaction_count']} txns) between same parties",
                    }
                )

        return sorted(fraud_rings, key=lambda x: x["risk_score"], reverse=True)

    def get_network_statistics(self):
        """Get overall network statistics with simple, fast queries"""
        if not self.connected:
            return {}

        try:
            with self.driver.session() as session:
                # Simple, separate queries that should be fast
                sender_count = session.run(
                    "MATCH (s:Sender) RETURN count(s) as count"
                ).single()["count"]
                beneficiary_count = session.run(
                    "MATCH (b:Beneficiary) RETURN count(b) as count"
                ).single()["count"]
                relationship_count = session.run(
                    "MATCH ()-[r:SENT_TO]->() RETURN count(r) as count"
                ).single()["count"]

                # Skip more complex queries that might be slow
                return {
                    "total_senders": sender_count or 0,
                    "total_beneficiaries": beneficiary_count or 0,
                    "total_relationships": relationship_count or 0,
                    "total_network_amount": "Calculated during analysis",
                    "high_risk_nodes": "Calculated during detection",
                }
        except Exception as e:
            return {
                "total_senders": "N/A",
                "total_beneficiaries": "N/A",
                "total_relationships": "N/A",
                "total_network_amount": "N/A",
                "high_risk_nodes": "N/A",
            }

    def export_fraud_rings_for_visualization(self, fraud_rings: List[Dict]):
        """Export fraud ring data for visualization"""
        nodes = []
        edges = []
        node_ids = set()

        for i, ring in enumerate(fraud_rings[:5]):  # Top 5 rings
            if ring["type"] == "Hub Pattern":
                # Add hub node
                hub_id = f"hub_{ring['hub']}"
                if hub_id not in node_ids:
                    nodes.append(
                        {
                            "id": hub_id,
                            "label": ring["hub_name"][:20],
                            "type": "beneficiary",
                            "risk_score": ring["risk_score"],
                            "size": min(50, 20 + ring["sender_count"] * 2),
                            "color": (
                                "#FF6B6B" if ring["risk_score"] > 0.7 else "#4ECDC4"
                            ),
                        }
                    )
                    node_ids.add(hub_id)

                # Add sender nodes and edges
                for conn in ring["connections"][:10]:  # Limit to 10 connections
                    sender_id = f"sender_{conn['sender']}"
                    if sender_id not in node_ids:
                        nodes.append(
                            {
                                "id": sender_id,
                                "label": conn["sender"][:15],
                                "type": "sender",
                                "risk_score": conn["risk"],
                                "size": 20,
                                "color": "#95E1D3",
                            }
                        )
                        node_ids.add(sender_id)

                    edges.append(
                        {
                            "id": f"edge_{sender_id}_{hub_id}",
                            "from": sender_id,
                            "to": hub_id,
                            "label": f"${conn['amount']:,.0f}",
                            "width": max(1, min(10, conn["amount"] / 5000)),
                            "color": "#FF6B6B" if conn["risk"] > 0.6 else "#95E1D3",
                        }
                    )

        return {"nodes": nodes, "edges": edges}


def _clean_name(name):
    """Clean and standardize name for comparison"""
    if pd.isna(name) or not isinstance(name, str):
        return ""
    import re

    return re.sub(r"[^A-Za-z\s]", "", name.upper().strip())


def _extract_first_last_4(name):
    """Extract first 4 and last 4 characters from name"""
    if pd.isna(name) or not isinstance(name, str):
        return "", ""
    clean = _clean_name(name)
    if len(clean) < 8:
        return clean, clean
    return clean[:4], clean[-4:]


def _calculate_comprehensive_similarity(name1, name2):
    """Calculate comprehensive similarity score combining all methods"""
    if pd.isna(name1) or pd.isna(name2) or not name1 or not name2:
        return 0

    from thefuzz import fuzz

    # Basic fuzzy string similarity
    basic_similarity = fuzz.ratio(_clean_name(name1), _clean_name(name2))

    # Token-based similarities
    token_sort_sim = fuzz.token_sort_ratio(_clean_name(name1), _clean_name(name2))
    token_set_sim = fuzz.token_set_ratio(_clean_name(name1), _clean_name(name2))

    # Take the maximum similarity from all methods
    max_similarity = max(basic_similarity, token_sort_sim, token_set_sim)

    return max_similarity


def _find_similar_names(target_name, all_names, threshold=70):
    """Find all names similar to target name above threshold"""
    similar_names = []

    for name in all_names:
        if name != target_name:
            similarity = _calculate_comprehensive_similarity(target_name, name)
            if similarity >= threshold:
                similar_names.append({"name": name, "similarity": similarity})

    # Sort by similarity descending
    similar_names.sort(key=lambda x: x["similarity"], reverse=True)
    return similar_names


def create_beneficiary_consolidation_map(
    df: pd.DataFrame, similarity_threshold: float = 70.0
):
    """
    Create a mapping from original beneficiary names to consolidated names
    using fuzzy matching logic
    """
    # Get all unique beneficiary names
    all_beneficiaries = df["beneficiary_name"].dropna().unique()

    # Create consolidation mapping
    consolidation_map = {}
    processed_names = set()

    for target_name in all_beneficiaries:
        if target_name in processed_names:
            continue

        # Find similar names
        similar_names = _find_similar_names(
            target_name, all_beneficiaries, similarity_threshold
        )

        if similar_names:
            # Create a group with the target name as canonical
            canonical_name = target_name
            group = [target_name] + [item["name"] for item in similar_names]

            # Map all names in group to canonical name
            for name in group:
                consolidation_map[name] = canonical_name
                processed_names.add(name)
        else:
            # No similar names found, maps to itself
            consolidation_map[target_name] = target_name
            processed_names.add(target_name)

    return consolidation_map
