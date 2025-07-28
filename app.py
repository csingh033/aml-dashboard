import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from graph_analysis import (
    build_transaction_graph,
    detect_suspicious_hubs,
    find_cycles,
    visualize_graph,
)
import hashlib
from graph_rag import build_graph_from_df, extract_graph_context, format_rag_prompt
import os
from dotenv import load_dotenv
from pyvis.network import Network
import plotly.graph_objects as go
from thefuzz import fuzz
from itertools import combinations
import re

# Make hash_value globally available


def hash_value(val, length=10):
    if pd.isna(val):
        return ""
    return hashlib.sha256(str(val).encode()).hexdigest()[:length]

def partial_name(name):
    if isinstance(name, str) and len(name) > 8:
        return name[:4] + name[-4:]
    return name

# Beneficiary Name Matching Functions
def clean_name(name):
    """Clean and standardize name for comparison"""
    if pd.isna(name) or not isinstance(name, str):
        return ""
    return re.sub(r'[^A-Za-z\s]', '', name.upper().strip())

def extract_first_last_4(name):
    """Extract first 4 and last 4 characters from name"""
    if pd.isna(name) or not isinstance(name, str):
        return "", ""
    clean = clean_name(name)
    if len(clean) < 8:
        return clean, clean
    return clean[:4], clean[-4:]

def calculate_hashed_similarity(name1, name2):
    """Calculate similarity for hashed names (same first 4 and last 4 letters)"""
    first1, last1 = extract_first_last_4(name1)
    first2, last2 = extract_first_last_4(name2)
    
    if first1 == first2 and last1 == last2 and first1 and last1:
        # High similarity if first 4 and last 4 match
        len_diff = abs(len(clean_name(name1)) - len(clean_name(name2)))
        # Penalize large length differences
        length_penalty = min(len_diff * 5, 30)
        return max(90 - length_penalty, 60)
    return 0

def calculate_word_order_similarity(name1, name2):
    """Calculate similarity for names with different word order"""
    words1 = set(clean_name(name1).split())
    words2 = set(clean_name(name2).split())
    
    if not words1 or not words2:
        return 0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0
    
    jaccard_similarity = (intersection / union) * 100
    
    # Boost if all words match but in different order
    if words1 == words2:
        return 95
    
    return jaccard_similarity

def calculate_partial_name_similarity(name1, name2):
    """Calculate similarity for partial names (full vs first+last, etc.)"""
    words1 = clean_name(name1).split()
    words2 = clean_name(name2).split()
    
    if not words1 or not words2:
        return 0
    
    # Check if shorter name is subset of longer name
    if len(words1) > len(words2):
        longer, shorter = words1, words2
    else:
        longer, shorter = words2, words1
    
    matches = sum(1 for word in shorter if word in longer)
    
    if matches == len(shorter) and len(shorter) >= 2:
        # All words in shorter name found in longer name
        return 85
    elif matches == len(shorter) and len(shorter) == 1:
        # Single word match (less reliable)
        return 60
    
    return (matches / max(len(words1), len(words2))) * 70

def calculate_comprehensive_similarity(name1, name2):
    """Calculate comprehensive similarity score combining all methods"""
    if pd.isna(name1) or pd.isna(name2) or not name1 or not name2:
        return 0
    
    # Basic fuzzy string similarity
    basic_similarity = fuzz.ratio(clean_name(name1), clean_name(name2))
    
    # Different matching approaches
    hashed_sim = calculate_hashed_similarity(name1, name2)
    word_order_sim = calculate_word_order_similarity(name1, name2)
    partial_sim = calculate_partial_name_similarity(name1, name2)
    
    # Token-based similarities (more conservative for AML)
    token_sort_sim = fuzz.token_sort_ratio(clean_name(name1), clean_name(name2))
    
    # Make token_set_ratio more conservative by requiring better overall match
    token_set_sim = fuzz.token_set_ratio(clean_name(name1), clean_name(name2))
    
    # AML-specific conservative adjustments for token_set_ratio
    words1 = clean_name(name1).split()
    words2 = clean_name(name2).split()
    set1 = set(words1)
    set2 = set(words2)
    
    # Critical fix: If one set is a proper subset of another, limit similarity
    if set1 != set2 and (set1.issubset(set2) or set2.issubset(set1)):
        # For AML, subset matches should not exceed 75% unless they're very similar
        if token_set_sim > 75:
            # Calculate overlap ratio - how much of the larger set is covered
            intersection = len(set1.intersection(set2))
            larger_set_size = max(len(set1), len(set2))
            overlap_ratio = intersection / larger_set_size if larger_set_size > 0 else 0
            
            # Adjust token_set_sim based on overlap ratio
            max_subset_similarity = 50 + (overlap_ratio * 25)  # Max 75% for perfect overlap
            token_set_sim = min(token_set_sim, max_subset_similarity)
    
    # For AML purposes, be more conservative with token_set_ratio
    # If token_set gives high score but basic similarity is low, reduce it
    if token_set_sim > 80 and basic_similarity < 60:
        token_set_sim = min(token_set_sim, basic_similarity + 20)
    
    # Additional check: if names have very different lengths, be more conservative
    len1, len2 = len(clean_name(name1)), len(clean_name(name2))
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # If length difference is significant, apply penalty to token-based matches
    if length_ratio < 0.6:  # Names differ significantly in length
        token_set_sim = min(token_set_sim, 75)
        token_sort_sim = min(token_sort_sim, 75)
    
    # If one name has significantly fewer words, be more conservative
    if len(words1) != len(words2) and abs(len(words1) - len(words2)) > 1:
        if token_set_sim > 90:
            token_set_sim = min(token_set_sim, 70)  # Cap at 70% for mismatched word counts
    
    # Special case: single word vs multi-word should not exceed 70%
    if (len(words1) == 1 and len(words2) > 1) or (len(words2) == 1 and len(words1) > 1):
        max_single_word_match = 70
        token_set_sim = min(token_set_sim, max_single_word_match)
        token_sort_sim = min(token_sort_sim, max_single_word_match)
    
    # Take the maximum similarity from all methods, but be more conservative
    max_similarity = max(basic_similarity, hashed_sim, word_order_sim, partial_sim, token_sort_sim, token_set_sim)
    
    return max_similarity

def find_similar_names(target_name, all_names, threshold=70):
    """Find all names similar to target name above threshold"""
    similar_names = []
    
    for name in all_names:
        if name != target_name:
            similarity = calculate_comprehensive_similarity(target_name, name)
            if similarity >= threshold:
                similar_names.append({
                    'name': name,
                    'similarity': similarity
                })
    
    # Sort by similarity descending
    similar_names.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_names

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(layout="wide")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 EDA",
        "💼 AML Dashboard",
        "👥 Beneficiary Analysis",
        "🤖 LLM Investigator",
        "🔍 Customer ID Lookup",
        "ℹ️ Model Information",
    ]
)

with tab1:
    st.title("📊 Exploratory Data Analysis (EDA)")
    uploaded_file = st.file_uploader("Upload your transaction CSV file", type="csv")
    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        try:
            uploaded_file.seek(0)
            # Read CSV with customer_no as string to prevent comma formatting
            df = pd.read_csv(uploaded_file, dtype={"customer_no": str})

            # Check if required columns exist for EDA
            required_columns = ["customer_no"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info(
                    """
                **Required CSV format:**
                Your CSV file must contain at least the following columns:
                - `customer_no`: Customer identification number
                
                **Optional columns:**
                - `CustomerName`: Customer name
                - `beneficiary_name`: Beneficiary name
                - `amount`: Transaction amount
                - `transfer_type`: Type of transfer
                - `createdDateTime`: Transaction timestamp
                """
                )
                st.stop()

            df["createdDateTime"] = pd.to_datetime(df["createdDateTime"])
            df["date"] = df["createdDateTime"].dt.date
            df["customer_no_hashed"] = df["customer_no"]

            # Apply partial_name to CustomerName and beneficiary_name
            if "CustomerName" in df.columns:
                df["customer_name_partial"] = df["CustomerName"].apply(partial_name)
            else:
                df["CustomerName"] = "Unknown"
                df["customer_name_partial"] = "Unknown"

            if "beneficiary_name" in df.columns:
                df["beneficiary_name_partial"] = df["beneficiary_name"].apply(partial_name)
            else:
                df["beneficiary_name"] = "Unknown"
                df["beneficiary_name_partial"] = "Unknown"

            df["is_self_transfer"] = df["customer_name_partial"] == df["beneficiary_name_partial"]

            st.session_state["processed_df"] = df

            # 1. Day-wise count of transactions by transfer type
            count_df = df.groupby(["date", "transfer_type"]).size().reset_index()
            count_df = count_df.rename(columns={0: "txn_count"})
            # 2. Day-wise sum of transactions by transfer type
            sum_df = df.groupby(["date", "transfer_type"])["amount"].sum().reset_index()
            sum_df = sum_df.rename(columns={"amount": "txn_sum"})
            # 3. Day-wise number of unique users by transfer type (as sender)
            user_df = (
                df.groupby(["date", "transfer_type"])["customer_no_hashed"]
                .nunique()
                .reset_index()
            )
            user_df = user_df.rename(columns={"customer_no_hashed": "unique_users"})
            import plotly.express as px

            st.subheader("Day-wise Count of Transactions by Transfer Type")
            fig1 = px.line(
                count_df,
                x="date",
                y="txn_count",
                color="transfer_type",
                markers=True,
                render_mode="svg",
            )
            fig1.update_traces(mode="lines+markers")
            st.plotly_chart(fig1, use_container_width=True)
            st.subheader("Day-wise Sum of Transactions by Transfer Type")
            fig2 = px.line(
                sum_df,
                x="date",
                y="txn_sum",
                color="transfer_type",
                markers=True,
                render_mode="svg",
            )
            fig2.update_traces(mode="lines+markers")
            st.plotly_chart(fig2, use_container_width=True)
            st.subheader("Day-wise Number of Unique Users by Transfer Type")
            fig3 = px.line(
                user_df,
                x="date",
                y="unique_users",
                color="transfer_type",
                markers=True,
                render_mode="svg",
            )
            fig3.update_traces(mode="lines+markers")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
    else:
        st.info("Upload a transaction CSV file to see EDA charts.")

with tab2:
    st.title("🕵️ AML Detection Dashboard")
    uploaded_file = st.session_state.get("uploaded_file", None)
    if uploaded_file is not None:
        try:
            df = st.session_state.get("processed_df", None)
            if df is None:
                st.info("Please upload a transaction CSV file in the EDA tab first.")
                st.stop()

            # Check if required columns exist
            required_columns = [
                "customer_no",
                "amount",
                "transfer_type",
                "createdDateTime",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info(
                    """
                **Required CSV format:**
                Your CSV file must contain at least the following columns:
                - `customer_no`: Customer identification number
                - `amount`: Transaction amount
                - `transfer_type`: Type of transfer
                - `createdDateTime`: Transaction timestamp
                
                **Optional columns:**
                - `CustomerName`: Customer name
                - `beneficiary_name`: Beneficiary name
                - `reference_no`: Transaction reference number
                """
                )
                st.stop()

            # Check if optional columns exist and create placeholders if missing
            if "reference_no" not in df.columns:
                df["reference_no"] = range(
                    len(df)
                )  # Create sequential reference numbers

            if "CustomerName" not in df.columns:
                df["CustomerName"] = "Unknown"

            if "beneficiary_name" not in df.columns:
                df["beneficiary_name"] = "Unknown"

            # Hash customer name and number for privacy
            df["customer_no_hashed"] = df["customer_no"].apply(hash_value)

            # Check if CustomerName column exists, if not create a placeholder
            if "CustomerName" in df.columns:
                df["CustomerName_hashed"] = df["CustomerName"].apply(hash_value)
            else:
                df["CustomerName"] = "Unknown"
                df["CustomerName_hashed"] = "Unknown"

            # Check if beneficiary_name column exists, if not create a placeholder
            if "beneficiary_name" in df.columns:
                df["beneficiary_name_hashed"] = df["beneficiary_name"].apply(hash_value)
            else:
                df["beneficiary_name"] = "Unknown"
                df["beneficiary_name_hashed"] = "Unknown"

            # Preprocessing
            df["createdDateTime"] = pd.to_datetime(df["createdDateTime"])
            df["hour"] = df["createdDateTime"].dt.hour
            df["day_of_week"] = df["createdDateTime"].dt.dayofweek
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            df["amount"] = df["amount"].fillna(0)
            df["is_international"] = df["transfer_type"].apply(
                lambda x: 1 if x == "INTERNATIONAL_PAYMENT" else 0
            )
            df["has_beneficiary"] = df["beneficiary_name"].notna().astype(int)
            df["transaction_count"] = df.groupby("customer_no")[
                "reference_no"
            ].transform("count")
            df["unique_beneficiaries"] = df.groupby("customer_no")[
                "beneficiary_name"
            ].transform("nunique")
            df["date"] = df["createdDateTime"].dt.date

            # --- Anomaly Detection (Moved before suspects filtering) ---
            features = df[
                [
                    "amount",
                    "hour",
                    "day_of_week",
                    "is_international",
                    "has_beneficiary",
                    "transaction_count",
                    "unique_beneficiaries",
                ]
            ]

            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)

            # Anomaly Detection Configuration
            st.sidebar.subheader("🔧 Anomaly Detection Settings")
            contamination_rate = st.sidebar.slider(
                "Contamination Rate", 
                min_value=0.01, 
                max_value=0.20, 
                value=0.02,  # Changed from 0.05 to 0.02 (more conservative)
                step=0.01,
                help="Expected proportion of anomalies in the data (lower = more conservative)"
            )
            
            # Anomaly Detection
            model = IsolationForest(contamination=contamination_rate, random_state=42)
            df["anomaly"] = model.fit_predict(scaled)
            df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)
            
            # Show detection statistics
            total_transactions = len(df)
            flagged_transactions = df["anomaly"].sum()
            st.info(f"🔍 Isolation Forest detected {flagged_transactions} anomalous transactions out of {total_transactions} total ({flagged_transactions/total_transactions*100:.1f}%)")

            # --- Customer-level summary ---
            # Calculate z-score for amount (already scaled in StandardScaler)
            suspects = df[df["anomaly"] == 1].copy()

            try:
                if not suspects.empty and isinstance(suspects, pd.DataFrame):
                    suspects["customer_no_hashed"] = suspects["customer_no"].apply(
                        hash_value
                    )

                    # Check if CustomerName column exists
                    if "CustomerName" in suspects.columns:
                        suspects["CustomerName_hashed"] = suspects[
                            "CustomerName"
                        ].apply(hash_value)
                    else:
                        suspects["CustomerName"] = "Unknown"
                        suspects["CustomerName_hashed"] = "Unknown"

                    # Check if beneficiary_name column exists
                    if "beneficiary_name" in suspects.columns:
                        suspects["beneficiary_name_hashed"] = suspects[
                            "beneficiary_name"
                        ].apply(hash_value)
                    else:
                        suspects["beneficiary_name"] = "Unknown"
                        suspects["beneficiary_name_hashed"] = "Unknown"

                    # Calculate z-score for amount (already scaled in StandardScaler)
                    suspects_features = suspects[
                        [
                            "amount",
                            "hour",
                            "day_of_week",
                            "is_international",
                            "has_beneficiary",
                            "transaction_count",
                            "unique_beneficiaries",
                        ]
                    ]
                    suspects["amount_zscore"] = pd.Series(
                        scaler.transform(suspects_features)[:, 0].ravel(),
                        index=suspects.index,
                    )
                    suspects["amount_percentile"] = (
                        pd.Series(suspects["amount"]).rank(pct=True) * 100
                    )

                    # Generate reasons for flagged transactions
                    def generate_reason(row):
                        reasons = []
                        if row["amount"] > df["amount"].quantile(0.95):
                            reasons.append("High amount")
                        if row["is_international"] == 1:
                            reasons.append("International transfer")
                        if row["has_beneficiary"] == 0:
                            reasons.append("No beneficiary")
                        if row["transaction_count"] > df["transaction_count"].quantile(
                            0.95
                        ):
                            reasons.append("High transaction frequency")
                        if row["unique_beneficiaries"] > df[
                            "unique_beneficiaries"
                        ].quantile(0.95):
                            reasons.append("Multiple beneficiaries")
                        if row["hour"] < 6 or row["hour"] > 22:
                            reasons.append("Unusual timing")
                        if not reasons:
                            reasons.append("Anomalous pattern")
                        return " | ".join(reasons)

                    suspects["reason"] = suspects.apply(generate_reason, axis=1)

                    def consolidate_reasons(reasons):
                        return " | ".join(sorted(set(reasons)))

                    def consolidate_types(types):
                        return ", ".join(sorted(set(types)))

                    def consolidate_beneficiaries(beneficiaries):
                        return ", ".join(
                            sorted(set([str(b) for b in beneficiaries if pd.notna(b)]))
                        )

                    customer_summary = (
                        suspects.groupby(["customer_no_hashed", "CustomerName_hashed"])
                        .agg(
                            total_flagged_txns=("amount", "count"),
                            total_flagged_amount=("amount", "sum"),
                            transfer_types=("transfer_type", consolidate_types),
                            beneficiaries=(
                                "beneficiary_name_hashed",
                                consolidate_beneficiaries,
                            ),
                            reasons=("reason", consolidate_reasons),
                            max_zscore=("amount_zscore", "max"),
                            max_percentile=("amount_percentile", "max"),
                        )
                        .reset_index()
                    )

                    # Add total transactions for each customer
                    total_txns_by_customer = (
                        df.groupby("customer_no_hashed")["reference_no"]
                        .count()
                        .reset_index()
                    )
                    total_txns_by_customer.columns = ["customer_no_hashed", "total_txn"]
                    customer_summary = customer_summary.merge(
                        total_txns_by_customer, on="customer_no_hashed", how="left"
                    )

                    def make_story(row):
                        story = f"Customer {row['CustomerName_hashed']} (#{row['customer_no_hashed']}) had {row['total_flagged_txns']} flagged transactions totaling {row['total_flagged_amount']}. "
                        story += f"Transfer types: {row['transfer_types']}. Beneficiaries: {row['beneficiaries']}. "
                        story += f"Criticality: z-score={row['max_zscore']:.2f}, percentile={row['max_percentile']:.1f}."
                        return story

                    customer_summary["story"] = customer_summary.apply(
                        make_story, axis=1
                    )

                    # Add filtering controls for high-risk customers
                    st.subheader("🧑‍💼 Customer-level AML Summary")
                    
                    # Create filter controls
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        min_flagged_txns = st.number_input(
                            "Minimum flagged transactions", 
                            min_value=1, 
                            value=4,  # Increased from 2 to 4
                            help="Show customers with at least this many flagged transactions"
                        )
                    
                    with col2:
                        min_zscore = st.number_input(
                            "Minimum Z-score threshold", 
                            min_value=0.0, 
                            value=2.0,  # Increased from 1.5 to 2.0
                            step=0.1,
                            help="Show customers with max Z-score above this threshold"
                        )
                    
                    with col3:
                        min_percentile = st.number_input(
                            "Minimum percentile threshold", 
                            min_value=0.0, 
                            max_value=100.0, 
                            value=90.0,  # Increased from 80.0 to 90.0
                            step=5.0,
                            help="Show customers with max percentile above this threshold"
                        )
                    
                    # Add additional filter row for more restrictive filtering
                    col4, col5, col6 = st.columns(3)
                    
                    with col4:
                        min_flagged_amount = st.number_input(
                            "Minimum total flagged amount", 
                            min_value=0.0, 
                            value=10000.0,  # New filter for high-value customers
                            step=1000.0,
                            help="Show customers with total flagged amount above this threshold"
                        )
                    
                    with col5:
                        min_risk_score = st.number_input(
                            "Minimum risk score", 
                            min_value=0.0, 
                            value=1.5,  # New composite risk filter
                            step=0.1,
                            help="Show customers with composite risk score above this threshold"
                        )
                        
                    with col6:
                        max_results = st.number_input(
                            "Max customers to show", 
                            min_value=1, 
                            max_value=100, 
                            value=25,  # Limit to top 25 customers
                            help="Limit results to top N highest-risk customers"
                        )
                    
                    # Apply filtering criteria (more restrictive)
                    filtered_summary = customer_summary[
                        (customer_summary["total_flagged_txns"] >= min_flagged_txns) &
                        (customer_summary["max_zscore"] >= min_zscore) &
                        (customer_summary["max_percentile"] >= min_percentile) &
                        (customer_summary["total_flagged_amount"] >= min_flagged_amount)
                    ].copy()
                    
                    # Sort by risk level (combination of z-score and percentile)
                    filtered_summary["risk_score"] = (
                        filtered_summary["max_zscore"] * 0.6 + 
                        filtered_summary["max_percentile"] * 0.004  # Scale percentile to similar range
                    )
                    filtered_summary = filtered_summary.sort_values("risk_score", ascending=False)
                    
                    # Apply additional risk score filter and limit results
                    filtered_summary = filtered_summary[
                        filtered_summary["risk_score"] >= min_risk_score
                    ].head(max_results)

                    # Create a list of columns to display for customer summary, checking if they exist
                    summary_display_columns = []
                    if "CustomerName" in filtered_summary.columns:
                        summary_display_columns.extend(
                            ["CustomerName", "CustomerName_hashed"]
                        )
                    summary_display_columns.extend(
                        [
                            "customer_no_hashed",
                            "total_txn",
                            "total_flagged_txns",
                            "total_flagged_amount",
                            "max_zscore",
                            "max_percentile",
                            "risk_score",
                            "transfer_types",
                            "beneficiaries",
                            "reasons",
                            "story",
                        ]
                    )

                    # Display results with filtering info
                    if len(filtered_summary) > 0:
                        st.success(f"✅ Found {len(filtered_summary)} high-risk customers out of {len(customer_summary)} total flagged customers")
                        st.dataframe(
                            filtered_summary[summary_display_columns],
                            use_container_width=True,
                        )
                        
                        # Add download option for filtered results
                        csv_data = filtered_summary[summary_display_columns].to_csv(index=False)
                        st.download_button(
                            label="📥 Download High-Risk Customers CSV",
                            data=csv_data,
                            file_name=f"high_risk_customers_{min_zscore}z_{min_percentile}p.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"⚠️ No customers meet the criteria (min {min_flagged_txns} transactions, Z-score ≥ {min_zscore}, percentile ≥ {min_percentile}%, amount ≥ ${min_flagged_amount:,.0f}, risk score ≥ {min_risk_score})")
                        st.info(f"💡 Total customers with any flagged transactions: {len(customer_summary)}")
                        
                        # Show summary statistics
                        if len(customer_summary) > 0:
                            st.write("**Summary of all flagged customers:**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Flagged Customers", len(customer_summary))
                            with col2:
                                st.metric("Max Z-score", f"{customer_summary['max_zscore'].max():.2f}")
                            with col3:
                                st.metric("Max Percentile", f"{customer_summary['max_percentile'].max():.1f}%")
                            with col4:
                                st.metric("Max Amount", f"${customer_summary['total_flagged_amount'].max():,.0f}")
                            
                            # Show distribution
                            st.write("**💡 Try lowering the filter thresholds above to see more customers**")

                else:
                    st.subheader("🧑‍💼 Customer-level AML Summary")
                    st.info("No anomalous transactions detected.")

            except Exception as e:
                st.error(f"❌ Error in customer summary processing: {str(e)}")
                import traceback

                st.code(traceback.format_exc())

            # --- Customer-centric Transaction Network Graph ---
            st.subheader("🔎 Customer Transaction Network Explorer")
            st.markdown(
                "Enter a customer number below to view only their transaction network and any cycles they are involved in. This will help you focus on individual customer behavior and connections."
            )
            customer_input = st.text_input(
                "Enter a customer number to view their transaction network and cycles:"
            )
            if customer_input:
                # Filter transactions where the hashed customer is sender or receiver
                cust_mask = (df["customer_no_hashed"] == customer_input) | (
                    df["beneficiary_name_hashed"] == customer_input
                )
                df_cust = df[cust_mask].copy()  # ensure DataFrame

                # Create a list of columns to display, checking if they exist
                display_columns = ["customer_no", "customer_no_hashed"]
                if "CustomerName" in df_cust.columns:
                    display_columns.extend(["CustomerName", "CustomerName_hashed"])
                display_columns.extend(["transfer_type"])
                if "beneficiary_name" in df_cust.columns:
                    display_columns.extend(
                        ["beneficiary_name", "beneficiary_name_hashed"]
                    )
                display_columns.extend(["amount", "createdDateTime"])

                st.dataframe(df_cust[display_columns], use_container_width=True)
                topup_count = (
                    df_cust["transfer_type"].astype(str).str.upper().eq("TOP-UP").sum()
                )
                if not df_cust.empty:
                    # Build graph using hashed values (all transactions where customer is sender or receiver)
                    def build_hashed_graph(df):
                        G = nx.DiGraph()
                        for _, row in df.iterrows():
                            sender = row["customer_no_hashed"]
                            receiver = row["beneficiary_name_hashed"]
                            # If beneficiary is missing and transfer_type is Top-up, use 'TOP-UP' node
                            if (pd.isna(receiver) or receiver == "") and str(
                                row.get("transfer_type", "")
                            ).upper() == "TOP-UP":
                                receiver = "TOP-UP"
                            if pd.isna(receiver) or receiver == "":
                                continue
                            G.add_node(sender, type="customer")
                            G.add_node(receiver, type="beneficiary")
                            G.add_edge(
                                sender,
                                receiver,
                                amount=row["amount"],
                                transfer_type=row["transfer_type"],
                                created=row["createdDateTime"],
                                reference_no=row["reference_no"],
                            )
                        return G

                    G_cust = build_hashed_graph(df_cust)
                    # Highlight the customer node
                    if customer_input in G_cust.nodes:
                        hubs = [customer_input]
                    else:
                        hubs = []
                    # Calculate total transactions for each node (sender or receiver) using a flattened Series from df_cust
                    all_nodes = pd.concat(
                        [
                            pd.Series(df_cust["customer_no_hashed"]),
                            pd.Series(df_cust["beneficiary_name_hashed"]),
                        ]
                    ).value_counts()
                    # Special handling for TOP-UP node: count edges to 'TOP-UP'
                    topup_count = df_cust[
                        df_cust["transfer_type"].str.upper() == "TOP-UP"
                    ].shape[0]

                    def node_label(node):
                        if node == "TOP-UP":
                            # Count edges where receiver is 'TOP-UP' in df_cust
                            topup_count = sum(
                                (
                                    pd.isna(row["beneficiary_name_hashed"])
                                    or row["beneficiary_name_hashed"] == ""
                                )
                                and str(row.get("transfer_type", "")).upper()
                                == "TOP-UP"
                                for _, row in df_cust.iterrows()
                            )
                            return f"TOP-UP ({topup_count})"
                        count = all_nodes.get(node, 0)
                        return f"{node} ({count})"

                    # Build label mapping for networkx
                    labels = {n: node_label(n) for n in G_cust.nodes}
                    # --- Pie chart for transaction type distribution ---
                    import plotly.express as px

                    type_counts = pd.Series(df_cust["transfer_type"]).value_counts()
                    type_sums = df_cust.groupby("transfer_type")["amount"].sum()

                    col1, col2 = st.columns(2)
                    with col1:
                        fig_pie_counts = px.pie(
                            type_counts,
                            values=type_counts.values,
                            names=type_counts.index,
                            title="Transaction Count by Type",
                        )
                        st.plotly_chart(fig_pie_counts, use_container_width=True)

                    with col2:
                        fig_pie_sums = px.pie(
                            type_sums,
                            values=type_sums.values,
                            names=type_sums.index,
                            title="Transaction Sum by Type",
                        )
                        st.plotly_chart(fig_pie_sums, use_container_width=True)
                    # --- Timeline histogram for transaction types (amount over time, day-wise) ---
                    df_cust["date"] = pd.to_datetime(df_cust["createdDateTime"]).dt.date
                    agg = (
                        df_cust.groupby(["date", "transfer_type"])
                        .agg(sum_amount=("amount", "sum"), count=("amount", "count"))
                        .reset_index()
                    )
                    import plotly.express as px

                    fig_hist = px.bar(
                        agg,
                        x="date",
                        y="sum_amount",
                        color="transfer_type",
                        barmode="stack",
                        title="Daily Transaction Sums and Counts by Type",
                        labels={"date": "Date", "sum_amount": "Sum of Amount"},
                        hover_data={
                            "count": True,
                            "sum_amount": True,
                            "transfer_type": True,
                        },
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                    # --- Improved cycle detection: run on the full customer subgraph ---
                    cycles = find_cycles(G_cust)
                    st.markdown(
                        f"**Cycles involving this customer:** {cycles if cycles else 'None found'}"
                    )

                    # Create interactive network graph using Plotly
                    import plotly.graph_objects as go

                    # Prepare node positions using spring layout
                    pos = nx.spring_layout(G_cust, seed=42)

                    # Create node traces
                    node_x = []
                    node_y = []
                    node_text = []
                    node_color = []
                    node_size = []

                    # Calculate transaction counts and sums for each node
                    node_stats = {}
                    for node in G_cust.nodes():
                        if node == "TOP-UP":
                            # Count TOP-UP transactions
                            topup_txns = df_cust[
                                df_cust["transfer_type"].astype(str).str.upper()
                                == "TOP-UP"
                            ].shape[0]
                            topup_sum = df_cust[
                                df_cust["transfer_type"].astype(str).str.upper()
                                == "TOP-UP"
                            ]["amount"].sum()
                            node_stats[node] = {
                                "count": topup_txns,
                                "sum": topup_sum,
                                "name": "TOP-UP",
                            }
                        else:
                            # Count transactions where this node is sender or receiver
                            sender_txns = df_cust[df_cust["customer_no_hashed"] == node]
                            receiver_txns = df_cust[
                                df_cust["beneficiary_name_hashed"] == node
                            ]
                            total_txns = len(sender_txns) + len(receiver_txns)
                            total_sum = (
                                sender_txns["amount"].sum()
                                + receiver_txns["amount"].sum()
                            )

                            # Get name: prefer CustomerName if sender, else beneficiary_name if receiver
                            name = None
                            if (
                                not sender_txns.empty
                                and "CustomerName" in sender_txns.columns
                            ):
                                name_candidates = (
                                    sender_txns["CustomerName"].dropna().unique()
                                )
                                if (
                                    len(name_candidates) > 0
                                    and name_candidates[0] != "Unknown"
                                ):
                                    name = name_candidates[0]
                            if (
                                (not name or name == "Unknown")
                                and not receiver_txns.empty
                                and "beneficiary_name" in receiver_txns.columns
                            ):
                                ben_candidates = (
                                    receiver_txns["beneficiary_name"].dropna().unique()
                                )
                                if (
                                    len(ben_candidates) > 0
                                    and ben_candidates[0] != "Unknown"
                                ):
                                    name = ben_candidates[0]
                            if not name or name == "Unknown":
                                name = "Unknown"
                            node_stats[node] = {
                                "count": total_txns,
                                "sum": total_sum,
                                "name": name,
                            }

                    for node in G_cust.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)

                        # Enhanced node label with name, count, and sum
                        stats = node_stats[node]
                        if node == "TOP-UP":
                            node_text.append(
                                f"TOP-UP<br>{stats['count']} txns<br>${stats['sum']:,.0f}"
                            )
                        else:
                            node_text.append(
                                f"{stats['name']}<br>{stats['count']} txns<br>${stats['sum']:,.0f}"
                            )

                        # Color nodes based on type
                        if node == customer_input:
                            node_color.append("red")  # Highlight selected customer
                            node_size.append(30)
                        elif node == "TOP-UP":
                            node_color.append("orange")  # TOP-UP node
                            node_size.append(25)
                        else:
                            node_color.append("lightblue")  # Regular nodes
                            node_size.append(20)

                    # Create edge traces
                    edge_x = []
                    edge_y = []
                    edge_text = []

                    for edge in G_cust.edges(data=True):
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                        # Edge hover text
                        amount = edge[2].get("amount", "N/A")
                        transfer_type = edge[2].get("transfer_type", "N/A")
                        edge_text.append(f"Amount: {amount}<br>Type: {transfer_type}")

                    # Create the network graph
                    fig = go.Figure()

                    # Add edges
                    fig.add_trace(
                        go.Scatter(
                            x=edge_x,
                            y=edge_y,
                            mode="lines",
                            line=dict(width=1, color="gray"),
                            hoverinfo="none",
                            showlegend=False,
                        )
                    )

                    # Add nodes
                    fig.add_trace(
                        go.Scatter(
                            x=node_x,
                            y=node_y,
                            mode="markers+text",
                            marker=dict(
                                size=node_size,
                                color=node_color,
                                line=dict(width=2, color="white"),
                            ),
                            text=node_text,
                            textposition="middle center",
                            textfont=dict(size=8),
                            hoverinfo="text",
                            hovertext=node_text,
                            showlegend=False,
                        )
                    )

                    # Update layout for better interactivity
                    fig.update_layout(
                        title=f"Transaction Network for Customer {customer_input}",
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                        yaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                        plot_bgcolor="white",
                        height=600,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Add network analysis insights
                    st.subheader("🔍 Network Analysis Insights")

                    # Calculate network metrics
                    if len(G_cust.nodes) > 1:
                        # Node centrality
                        in_degree = dict(G_cust.in_degree())
                        out_degree = dict(G_cust.out_degree())

                        # Find most connected nodes with enhanced info
                        most_connected = sorted(
                            out_degree.items(), key=lambda x: x[1], reverse=True
                        )[:3]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Network Statistics:**")
                            st.write(f"• Total nodes: {len(G_cust.nodes)}")
                            st.write(f"• Total edges: {len(G_cust.edges)}")
                            st.write(f"• Network density: {nx.density(G_cust):.3f}")
                        with col2:
                            st.write("**Most Connected Nodes:**")
                            for node, degree in most_connected:
                                stats = node_stats[node]
                                if node == customer_input:
                                    st.write(
                                        f"• **{stats['name']}** ({degree} connections, {stats['count']} txns, ${stats['sum']:,.0f}) - *Selected*"
                                    )
                                else:
                                    st.write(
                                        f"• {stats['name']} ({degree} connections, {stats['count']} txns, ${stats['sum']:,.0f})"
                                    )
                        # Show cycles if found
                        if cycles:
                            st.write("**🔴 Suspicious Cycles Detected:**")
                            for i, cycle in enumerate(
                                cycles[:3], 1
                            ):  # Show first 3 cycles
                                cycle_with_names = []
                                for node in cycle:
                                    if node in node_stats:
                                        cycle_with_names.append(
                                            node_stats[node]["name"]
                                        )
                                    else:
                                        cycle_with_names.append(node)
                                st.write(f"{i}. {' → '.join(cycle_with_names)}")
                        else:
                            st.write("**✅ No suspicious cycles detected**")
                    else:
                        st.info("Network too small for detailed analysis.")
                else:
                    st.info("No transactions found for this customer.")
            else:
                st.info(
                    "Enter a customer number above to explore their transaction network."
                )
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
    else:
        st.info("Please upload a transaction CSV file in the EDA tab first.")



with tab3:
    st.header("👥 Beneficiary Analysis")
    st.markdown(
        """
    This section identifies beneficiaries who receive funds from multiple customers, which can be an indicator of mule activity or other financial irregularities.
    """
    )
    uploaded_file = st.session_state.get("uploaded_file", None)
    if uploaded_file is not None:
        try:
            df = st.session_state.get("processed_df", None)
            if df is None:
                st.info("Please upload a transaction CSV file in the EDA tab first.")
                st.stop()

            if "beneficiary_name" not in df.columns:
                st.warning("Beneficiary name column not found in the uploaded file.")
            else:
                # Add filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_txn = st.number_input("Minimum Transactions", 1, 100, 1)
                with col2:
                    min_unique_senders = st.number_input("Minimum Unique Senders", 1, 100, 1)
                with col3:
                    min_total_received_amount = st.number_input("Minimum Total Received Amount", 0.0, 10000000.0, 0.0, step=1000.0)

                # Handle nationality filtering with error checking
                try:
                    if 'nationality' in df.columns:
                        # Clean nationality data - convert all to strings and handle NaN
                        df['nationality'] = df['nationality'].fillna('Unknown').astype(str)
                        nationalities = sorted([str(n) for n in df['nationality'].unique() if str(n) != 'nan'])
                        if not nationalities:
                            nationalities = ['Unknown']
                    else:
                        # Create default nationality if column doesn't exist
                        df['nationality'] = 'Unknown'
                        nationalities = ['Unknown']
                    
                    selected_nationalities = st.multiselect("Nationalities", nationalities, default=nationalities)
                except Exception as e:
                    st.warning(f"Issue with nationality filtering: {e}. Using all data.")
                    df['nationality'] = 'Unknown'
                    nationalities = ['Unknown']
                    selected_nationalities = ['Unknown']

                # Filter original DataFrame based on selected nationalities
                try:
                    if selected_nationalities:
                        df_filtered_by_nationality = df[df['nationality'].isin(selected_nationalities)].copy()
                    else:
                        df_filtered_by_nationality = df.copy()
                except Exception as e:
                    st.warning(f"Filtering by nationality failed: {e}. Using all data.")
                    df_filtered_by_nationality = df.copy()

                # Beneficiary Analysis with improved error handling
                try:
                    # Step 1: Automatic name consolidation for whitespace/case variations
                    st.subheader("🔧 Automatic Name Consolidation")
                    
                    # Create normalized name mapping
                    def normalize_name(name):
                        """Normalize name by removing extra spaces and converting to uppercase"""
                        if pd.isna(name) or not isinstance(name, str):
                            return ""
                        # Remove extra spaces and convert to uppercase
                        return ' '.join(name.upper().split())
                    
                    # Create mapping of original names to normalized names
                    all_beneficiary_names = df_filtered_by_nationality['beneficiary_name'].dropna().unique()
                    name_groups = {}
                    normalized_to_canonical = {}
                    
                    # Group names by their normalized form
                    for name in all_beneficiary_names:
                        normalized = normalize_name(name)
                        if normalized:
                            if normalized not in name_groups:
                                name_groups[normalized] = []
                            name_groups[normalized].append(name)
                    
                    # Find groups with multiple variations and select canonical names
                    consolidated_groups = []
                    total_consolidations = 0
                    
                    for normalized, variations in name_groups.items():
                        if len(variations) > 1:
                            # Sort by length (prefer shorter, cleaner names) then alphabetically
                            canonical_name = sorted(variations, key=lambda x: (len(x), x))[0]
                            consolidated_groups.append({
                                'Canonical Name': canonical_name,
                                'Variations': variations,
                                'Count': len(variations)
                            })
                            total_consolidations += len(variations) - 1
                            
                            # Map all variations to canonical name
                            for variation in variations:
                                normalized_to_canonical[variation] = canonical_name
                        else:
                            # Single name, maps to itself
                            normalized_to_canonical[variations[0]] = variations[0]
                    
                    # Show consolidation summary
                    if consolidated_groups:
                        st.success(f"✅ Found {len(consolidated_groups)} groups with name variations. Consolidated {total_consolidations} duplicate entries.")
                        
                        # Show consolidation details in an expander
                        with st.expander(f"📋 View {len(consolidated_groups)} Consolidated Groups"):
                            consolidation_df = pd.DataFrame(consolidated_groups)
                            
                            # Create detailed view showing all variations
                            detailed_data = []
                            for group in consolidated_groups:
                                canonical = group['Canonical Name']
                                for i, variation in enumerate(group['Variations']):
                                    detailed_data.append({
                                        'Canonical Name': canonical if i == 0 else '',
                                        'Variation': variation,
                                        'Status': 'Main' if variation == canonical else 'Consolidated'
                                    })
                            
                            detailed_df = pd.DataFrame(detailed_data)
                            st.dataframe(detailed_df, use_container_width=True)
                    else:
                        st.info("ℹ️ No name variations found - all beneficiary names are already unique.")
                    
                    # Step 2: Apply consolidation to the data
                    df_consolidated = df_filtered_by_nationality.copy()
                    df_consolidated['beneficiary_name_original'] = df_consolidated['beneficiary_name']
                    df_consolidated['beneficiary_name'] = df_consolidated['beneficiary_name'].map(
                        lambda x: normalized_to_canonical.get(x, x)
                    )
                    
                    # Step 3: Create beneficiary summary with consolidated names
                    # Simplified aggregation to avoid complex lambda functions
                    beneficiary_summary = (
                        df_consolidated.groupby("beneficiary_name")
                        .agg(
                            total_received=("amount", "sum"),
                            transaction_count=("amount", "count"),
                            distinct_senders=("customer_no", "nunique"),
                        )
                        .reset_index()
                    )
                    
                    # Add self-transfer calculations separately to avoid complex lambda issues
                    if 'is_self_transfer' in df_consolidated.columns:
                        try:
                            self_transfers = df_consolidated[df_consolidated['is_self_transfer'] == True]
                            self_transfer_summary = (
                                self_transfers.groupby("beneficiary_name")
                                .agg(
                                    self_transfer_amount=("amount", "sum"),
                                    self_transfer_count=("amount", "count"),
                                )
                                .reset_index()
                            )
                            beneficiary_summary = beneficiary_summary.merge(
                                self_transfer_summary, 
                                on="beneficiary_name", 
                                how="left"
                            )
                            beneficiary_summary['self_transfer_amount'] = beneficiary_summary['self_transfer_amount'].fillna(0)
                            beneficiary_summary['self_transfer_count'] = beneficiary_summary['self_transfer_count'].fillna(0)
                        except Exception as e:
                            st.warning(f"Self-transfer calculation failed: {e}. Skipping self-transfer metrics.")
                            beneficiary_summary['self_transfer_amount'] = 0
                            beneficiary_summary['self_transfer_count'] = 0
                    else:
                        beneficiary_summary['self_transfer_amount'] = 0
                        beneficiary_summary['self_transfer_count'] = 0

                except Exception as e:
                    st.error(f"Error in beneficiary analysis: {e}")
                    st.stop()

                # Apply filters
                filtered_beneficiaries = beneficiary_summary[
                    (beneficiary_summary["transaction_count"] >= min_txn) &
                    (beneficiary_summary["distinct_senders"] >= min_unique_senders) &
                    (beneficiary_summary["total_received"] >= min_total_received_amount)
                ]

                st.subheader("Beneficiaries Receiving from Multiple Senders")
                st.dataframe(filtered_beneficiaries)

                # Further analysis on suspicious beneficiaries
                if not filtered_beneficiaries.empty:
                    selected_beneficiary = st.selectbox(
                        "Select a beneficiary to see transaction details:",
                        options=filtered_beneficiaries["beneficiary_name"],
                    )

                    if selected_beneficiary:
                        st.subheader(
                            f"Transaction Details for {selected_beneficiary}"
                        )
                        beneficiary_transactions = df_consolidated[
                            df_consolidated["beneficiary_name"] == selected_beneficiary
                        ]
                        
                        # Show both original and consolidated names
                        if 'beneficiary_name_original' in beneficiary_transactions.columns:
                            original_names = beneficiary_transactions['beneficiary_name_original'].unique()
                            if len(original_names) > 1:
                                st.info(f"📝 This consolidated beneficiary includes these name variations: {', '.join(original_names)}")
                        
                        st.dataframe(beneficiary_transactions)

                        # Charts with error handling
                        try:
                            st.subheader("Transaction Analysis")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Transaction Amount Over Time")
                                beneficiary_transactions['day_month'] = beneficiary_transactions['createdDateTime'].dt.strftime('%d %B')
                                daily_summary = beneficiary_transactions.groupby('day_month')['amount'].sum().reset_index()
                                # Create a sortable date column for correct chronological order
                                daily_summary['sort_date'] = pd.to_datetime(daily_summary['day_month'], format='%d %B')
                                daily_summary = daily_summary.sort_values(by='sort_date')
                                fig = px.bar(daily_summary, x="day_month", y="amount", title="Transaction Amount Over Time")
                                st.plotly_chart(fig)
                            with col2:
                                st.subheader("Total Amount by Nationality")
                                # Group by nationality and sum amount, also count transactions
                                if 'nationality' in beneficiary_transactions.columns:
                                    nationality_summary = beneficiary_transactions.groupby('nationality').agg(
                                        total_amount=('amount', 'sum'),
                                        transaction_count=('amount', 'count')
                                    ).reset_index()
                                    fig = px.pie(
                                        nationality_summary,
                                        values='total_amount',
                                        names='nationality',
                                        title='Total Amount by Nationality',
                                        hover_data=['transaction_count']
                                    )
                                    st.plotly_chart(fig)
                                else:
                                    st.info("Nationality data not available for this beneficiary.")
                        except Exception as e:
                            st.warning(f"Error creating charts: {e}")

                        # Cycle Detection with error handling
                        try:
                            st.subheader("Cyclic Transactions")
                            G = nx.from_pandas_edgelist(df_consolidated, "customer_no", "beneficiary_name", create_using=nx.DiGraph())
                            cycles = list(nx.simple_cycles(G))
                            beneficiary_cycles = [cycle for cycle in cycles if selected_beneficiary in cycle]
                            if beneficiary_cycles:
                                st.write("Found the following cycles:")
                                for cycle in beneficiary_cycles:
                                    st.write(cycle)
                            else:
                                st.write("No cycles found for this beneficiary.")
                        except Exception as e:
                            st.warning(f"Error in cycle detection: {e}")

                # New Beneficiary Name Matching & Consolidation Section
                st.markdown("---")
                st.subheader("🔍 Advanced Beneficiary Name Matching & Consolidation")
                st.markdown("""
                This section provides advanced fuzzy matching to identify beneficiaries with similar names beyond basic case/whitespace variations.
                It handles scenarios like hashed names, different word orders, and partial names.
                """)

                # Examples section
                with st.expander("📝 See Examples of Advanced Name Matching"):
                    st.markdown("""
                    **Advanced matching scenarios (beyond the automatic consolidation above):**
                    
                    **1. 🔒 Hashed/Masked Names:**
                    - `AAMA#########VEED` ➔ `AAMA#####VEED` ➔ `AAMAR NAVEED`
                    - *Matches based on first 4 and last 4 letters*
                    
                    **2. 🔄 Different Word Order:**
                    - `VICTOR JOHNNY IBANGA` ➔ `IBANGA VICTOR JOHNNY` ➔ `VICTOR IBANGA JOHNNY`
                    - *Same words in different sequence*
                    
                    **3. ✂️ Partial Names:**
                    - `VICTOR IBANGA JOHNNY` ➔ `VICTOR IBANGA` ➔ `VICTOR JOHNNY`
                    - *Full name vs first+last or first+middle combinations*
                    """)

                # Name matching controls with error handling
                try:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # Input for beneficiary name search (now using consolidated names)
                        all_beneficiary_names = [str(name) for name in df_consolidated['beneficiary_name'].unique() if pd.notna(name)]
                        search_input = st.text_input(
                            "🔎 Enter beneficiary name to find similar matches:",
                            placeholder="Type a beneficiary name...",
                            help="Enter a beneficiary name to find all similar variations"
                        )

                    with col2:
                        similarity_threshold = st.slider(
                            "Similarity Threshold (%)",
                            min_value=50,
                            max_value=95,
                            value=70,
                            step=1,
                            help="Adjust the similarity threshold - higher values require closer matches"
                        )

                    # Determine which name to search for
                    target_name = search_input.strip()

                    if target_name:
                        st.markdown(f"### 🎯 Finding matches for: **{target_name}**")
                        
                        # Find similar names
                        similar_names = find_similar_names(target_name, all_beneficiary_names, similarity_threshold)
                        
                        if similar_names:
                            # Create consolidated view
                            st.subheader("✅ Similar Names Found & Breakdown")
                            
                            # Prepare data for the consolidated table
                            consolidated_data = []
                            total_consolidated_amount = 0
                            total_consolidated_transactions = 0
                            total_consolidated_senders = set()
                            
                            # Include the target name itself
                            all_matching_names = [target_name] + [item['name'] for item in similar_names]
                            
                            # Calculate consolidated statistics and create merged table
                            merged_table_data = []
                            
                            # Add target name first
                            target_transactions = df_consolidated[df_consolidated['beneficiary_name'] == target_name]
                            if not target_transactions.empty:
                                target_total = target_transactions['amount'].sum()
                                target_count = len(target_transactions)
                                target_senders = set(target_transactions['customer_no'].unique())
                                
                                merged_table_data.append({
                                    'Beneficiary Name': target_name,
                                    'Similarity Score (%)': 100.0,
                                    'Match Type': 'Original',
                                    'Total Amount': f"${target_total:,.2f}",
                                    'Transaction Count': target_count,
                                    'Unique Senders': len(target_senders)
                                })
                                
                                total_consolidated_amount += target_total
                                total_consolidated_transactions += target_count
                                total_consolidated_senders.update(target_senders)
                            
                            # Add similar names
                            for item in similar_names:
                                name = item['name']
                                similarity = item['similarity']
                                name_transactions = df_consolidated[df_consolidated['beneficiary_name'] == name]
                                
                                if not name_transactions.empty:
                                    name_total = name_transactions['amount'].sum()
                                    name_count = len(name_transactions)
                                    name_senders = set(name_transactions['customer_no'].unique())
                                    
                                    merged_table_data.append({
                                        'Beneficiary Name': name,
                                        'Similarity Score (%)': similarity,
                                        'Match Type': 'Similar',
                                        'Total Amount': f"${name_total:,.2f}",
                                        'Transaction Count': name_count,
                                        'Unique Senders': len(name_senders)
                                    })
                                    
                                    total_consolidated_amount += name_total
                                    total_consolidated_transactions += name_count
                                    total_consolidated_senders.update(name_senders)
                                else:
                                    # Include names with no transactions for completeness
                                    merged_table_data.append({
                                        'Beneficiary Name': name,
                                        'Similarity Score (%)': similarity,
                                        'Match Type': 'Similar',
                                        'Total Amount': "$0.00",
                                        'Transaction Count': 0,
                                        'Unique Senders': 0
                                    })
                            
                            # Display merged table
                            merged_df = pd.DataFrame(merged_table_data)
                            # Sort by similarity score descending
                            merged_df = merged_df.sort_values('Similarity Score (%)', ascending=False)
                            st.dataframe(merged_df, use_container_width=True)
                            
                            # Display consolidated totals
                            st.subheader("🎯 Consolidated Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Total Consolidated Amount",
                                    f"${total_consolidated_amount:,.2f}",
                                    help="Sum of all amounts across all similar names"
                                )
                            
                            with col2:
                                st.metric(
                                    "Total Transactions",
                                    f"{total_consolidated_transactions:,}",
                                    help="Total number of transactions across all similar names"
                                )
                            
                            with col3:
                                st.metric(
                                    "Unique Senders",
                                    f"{len(total_consolidated_senders):,}",
                                    help="Number of unique senders across all similar names"
                                )
                            
                            with col4:
                                average_per_transaction = total_consolidated_amount / total_consolidated_transactions if total_consolidated_transactions > 0 else 0
                                st.metric(
                                    "Average per Transaction",
                                    f"${average_per_transaction:,.2f}",
                                    help="Average transaction amount across consolidated names"
                                )
                            
                            # Show all consolidated transactions
                            if st.checkbox("📋 Show All Consolidated Transactions"):
                                all_consolidated_transactions = df_consolidated[
                                    df_consolidated['beneficiary_name'].isin(all_matching_names)
                                ].copy()
                                
                                # Add similarity score column
                                all_consolidated_transactions['Name_Similarity_Score'] = all_consolidated_transactions['beneficiary_name'].apply(
                                    lambda x: 100.0 if x == target_name else next(
                                        (item['similarity'] for item in similar_names if item['name'] == x), 0
                                    )
                                )
                                
                                # Sort by similarity score descending, then by amount descending
                                all_consolidated_transactions = all_consolidated_transactions.sort_values(
                                    ['Name_Similarity_Score', 'amount'], ascending=[False, False]
                                )
                                
                                st.dataframe(all_consolidated_transactions, use_container_width=True)
                                
                                # Download option
                                csv_data = all_consolidated_transactions.to_csv(index=False)
                                st.download_button(
                                    label="💾 Download Consolidated Transactions",
                                    data=csv_data,
                                    file_name=f"consolidated_transactions_{target_name.replace(' ', '_')}.csv",
                                    mime="text/csv"
                                )
                        
                        else:
                            st.info(f"No similar names found for '{target_name}' with similarity threshold of {similarity_threshold}%")
                            st.markdown("💡 **Try lowering the similarity threshold or check spelling**")

                except Exception as e:
                    st.error(f"Error in name matching section: {e}")
                    st.markdown("💡 **Try refreshing the page or check your data format**")

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
            st.markdown("💡 **Please check your CSV file format and ensure all required columns are present**")
    else:
        st.info("Please upload a transaction CSV file in the EDA tab first.")

with tab5:
    st.markdown(
        """
    This tool uses a Retrieval-Augmented Generation (RAG) approach: it extracts transaction data for a customer and asks a Large Language Model (LLM) to analyze potential AML risks based on the transaction patterns.
    """
    )

    # OpenAI API Key input
    st.subheader("🔑 OpenAI API Configuration")
    api_key_input = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key to enable LLM analysis. You can get one from https://platform.openai.com/api-keys",
    )

    # Use input API key if provided, otherwise fall back to environment variable
    current_api_key = api_key_input if api_key_input else OPENAI_API_KEY

    uploaded_file = (
        st.session_state.get("uploaded_file", None)
        if "uploaded_file" in st.session_state
        else None
    )
    if uploaded_file is None:
        st.info("Please upload a transaction CSV file in the EDA tab first.")
    else:
        df = st.session_state.get("processed_df", None)
        if df is None:
            st.info("Please upload a transaction CSV file in the EDA tab first.")
            st.stop()
        # Hash customer numbers for privacy
        df["customer_no_hashed"] = df["customer_no"].apply(hash_value)
        df["beneficiary_name_hashed"] = df["beneficiary_name"].apply(hash_value)
        df["transfer_type_hashed"] = df["transfer_type"]

        # Check if CustomerName column exists and create hashed version
        if "CustomerName" in df.columns:
            df["CustomerName_hashed"] = df["CustomerName"].apply(hash_value)
        else:
            df["CustomerName"] = "Unknown"
            df["CustomerName_hashed"] = "Unknown"

        # Customer search input
        customer_options = sorted(df["customer_no_hashed"].unique())
        selected_customer = st.selectbox(
            "Enter or select a hashed customer number to investigate:",
            options=customer_options,
            index=None,
            placeholder="Type or select a hashed customer number...",
        )

        if selected_customer:
            # Extract all transactions involving the selected customer
            cust_mask = (df["customer_no_hashed"] == selected_customer) | (
                df["beneficiary_name_hashed"] == selected_customer
            )
            df_cust = df[cust_mask]

            if not df_cust.empty:
                # Prepare transaction data for RAG prompt
                transactions_data = []
                for _, row in df_cust.iterrows():
                    transaction_info = {
                        "customer_id_hash": row["customer_no_hashed"],
                        "transfer_type_hash": row["transfer_type_hashed"],
                        "beneficiary_id_hash": (
                            row["beneficiary_name_hashed"]
                            if pd.notna(row["beneficiary_name_hashed"])
                            else "N/A"
                        ),
                        "amount": row["amount"],
                        "datetime": row["createdDateTime"],
                    }
                    transactions_data.append(transaction_info)

                # Create RAG prompt
                prompt = f"""
                **AML Analysis Request for Customer: {selected_customer}**
                
                **Transaction Data:**
                {transactions_data}
                
                **Analysis Request:**
                Please provide a concise AML analysis for customer {selected_customer} with:
                
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

                # Create a list of columns to display for LLM analysis, checking if they exist
                llm_display_columns = ["customer_no", "customer_no_hashed"]
                if "CustomerName" in df_cust.columns:
                    llm_display_columns.extend(["CustomerName", "CustomerName_hashed"])
                llm_display_columns.extend(["transfer_type"])
                if "beneficiary_name" in df_cust.columns:
                    llm_display_columns.extend(
                        ["beneficiary_name", "beneficiary_name_hashed"]
                    )
                llm_display_columns.extend(["amount", "createdDateTime"])

                st.subheader("📊 Transaction Data for Analysis")
                st.dataframe(df_cust[llm_display_columns], use_container_width=True)

                st.subheader("🔍 RAG Prompt Sent to LLM:")
                st.code(prompt, language="markdown")

                # GPT-4o response
                if current_api_key:
                    import openai

                    try:
                        client = openai.OpenAI(api_key=current_api_key)
                        with st.spinner("Getting GPT-4o AML analysis..."):
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "You are an expert AML investigator. Provide concise, actionable analysis with clear risk assessment (HIGH/MEDIUM/LOW) and exactly 4 key bullet points. Use color coding: RED for HIGH risk, GREEN for LOW risk, BLUE for MEDIUM/NEUTRAL.",
                                        },
                                        {"role": "user", "content": prompt},
                                    ],
                                    max_tokens=800,
                                    temperature=0.2,
                                )
                                gpt_response = response.choices[0].message.content
                                st.subheader("🤖 LLM AML Analysis (GPT-4o):")

                                # Determine risk level and apply appropriate styling
                                risk_level = "MEDIUM"  # default
                                if (
                                    "HIGH" in gpt_response.upper()
                                    or "RED" in gpt_response.upper()
                                ):
                                    risk_level = "HIGH"
                                elif (
                                    "LOW" in gpt_response.upper()
                                    or "GREEN" in gpt_response.upper()
                                ):
                                    risk_level = "LOW"

                                # Apply appropriate styling based on risk level
                                if risk_level == "HIGH":
                                    st.error(gpt_response)
                                elif risk_level == "LOW":
                                    st.success(gpt_response)
                                else:  # MEDIUM
                                    st.info(gpt_response)
                            except openai.AuthenticationError:
                                st.error(
                                    "❌ Authentication failed. Please check your OpenAI API key."
                                )
                            except openai.RateLimitError:
                                st.error(
                                    "❌ Rate limit exceeded. Please try again later."
                                )
                            except openai.APIError as e:
                                st.error(f"❌ OpenAI API error: {e}")
                            except Exception as e:
                                st.error(f"❌ Error from OpenAI API: {e}")
                    except Exception as e:
                        st.error(f"❌ Error initializing OpenAI client: {e}")
                        st.info(
                            "💡 Try updating the requirements.txt with compatible versions of openai and httpx libraries."
                        )
                else:
                    st.warning(
                        "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file or enter it above."
                    )
            else:
                st.warning(
                    f"No transaction data found for customer {selected_customer}"
                )
        else:
            st.info("Please select a customer to analyze.")

with tab4:
    st.header("🔍 Customer ID Lookup")
    st.markdown(
        """
    Upload a CSV file containing customer IDs to get detailed information about customers.
    This tool helps investigators retrieve customer details for flagged transactions.
    
    **Required CSV format:** The file should have a column named 'customer_no_hashed' containing the customer numbers.
    """
    )

    # Check if we have processed data from the main dashboard
    if "uploaded_file" in st.session_state and "processed_df" not in st.session_state:
        # Process the original data and store it for lookup
        uploaded_file = st.session_state["uploaded_file"]
        uploaded_file.seek(0)
        # Read CSV with customer_no as string to prevent comma formatting
        df = pd.read_csv(uploaded_file, dtype={"customer_no": str})

        # Create a mapping of hashed to original customer numbers
        customer_mapping = {}
        for _, row in df.iterrows():
            customer_id = row["customer_no"]
            if customer_id not in customer_mapping:
                customer_mapping[customer_id] = {
                    "original_customer_no": row["customer_no"],
                    "original_customer_name": row["CustomerName"],
                    "total_transactions": len(
                        df[df["customer_no"] == row["customer_no"]]
                    ),
                }

        st.session_state["customer_mapping"] = customer_mapping
        st.session_state["processed_df"] = df
        st.success("✅ Customer mapping created from uploaded data!")

    # File upload for customer ID lookup
    lookup_file = st.file_uploader(
        "Upload CSV with hashed customer IDs", type="csv", key="lookup_uploader"
    )

    if lookup_file is not None:
        try:
            lookup_df = pd.read_csv(lookup_file)

            if "customer_no_hashed" not in lookup_df.columns:
                st.error(
                    "❌ The uploaded CSV must contain a column named 'customer_no_hashed'"
                )
            else:
                st.success(f"✅ Found {len(lookup_df)} hashed customer IDs to lookup")

                # Get customer mapping if available
                customer_mapping = st.session_state.get("customer_mapping", {})

                if not customer_mapping:
                    st.warning(
                        "⚠️ No customer mapping available. Please upload transaction data in the AML Dashboard tab first."
                    )
                else:
                    # Lookup results
                    results = []
                    found_count = 0

                    for _, row in lookup_df.iterrows():
                        customer_id = str(row["customer_no_hashed"])
                        if customer_id in customer_mapping:
                            mapping = customer_mapping[customer_id]
                            results.append(
                                {
                                    "customer_no": customer_id,
                                    "original_customer_no": mapping[
                                        "original_customer_no"
                                    ],
                                    "original_customer_name": mapping[
                                        "original_customer_name"
                                    ],
                                    "total_transactions": mapping["total_transactions"],
                                }
                            )
                            found_count += 1
                        else:
                            results.append(
                                {
                                    "customer_no": customer_id,
                                    "original_customer_no": "NOT FOUND",
                                    "original_customer_name": "NOT FOUND",
                                    "total_transactions": 0,
                                }
                            )

                    # Display results
                    results_df = pd.DataFrame(results)
                    st.subheader(
                        f"🔍 Lookup Results ({found_count}/{len(lookup_df)} found)"
                    )
                    st.dataframe(results_df, use_container_width=True)

                    # Download results
                    csv = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Lookup Results as CSV",
                        csv,
                        "customer_lookup_results.csv",
                        "text/csv",
                        key="download-lookup-csv",
                    )

                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Lookups", len(lookup_df))
                    with col2:
                        st.metric("Found", found_count)
                    with col3:
                        st.metric("Not Found", len(lookup_df) - found_count)

        except Exception as e:
            st.error(f"❌ Error processing the uploaded file: {str(e)}")
    else:
        st.info(
            "📁 Please upload a CSV file with hashed customer IDs to begin the lookup process."
        )

with tab6:
    st.markdown(
        """
    This AML Dashboard implements a comprehensive multi-layered approach to detect and analyze suspicious financial transactions. Here's an overview of our current methodology:

    **📊 1. Exploratory Data Analysis (EDA)**
    - **Daily Transaction Trends**: Three smooth line charts showing daily count, sum, and unique users by transfer type
    - **Pattern Recognition**: Identifies temporal patterns and transaction volume trends across different transfer types
    - **Data Quality Assessment**: Validates transaction data integrity and completeness

    **🕵️ 2. Anomaly Detection (Isolation Forest)**
    - **Unsupervised ML**: Uses Isolation Forest algorithm to detect unusual transactions without predefined rules
    - **Multi-dimensional Features**: Considers transaction amount, time patterns, international transfers, beneficiary presence, transaction frequency, and unique beneficiary count
    - **Adaptive Thresholds**: Automatically adjusts to data characteristics with 1% contamination rate
    - **Privacy-Preserving**: All customer identifiers are hashed using SHA-256 for data protection

    **🔍 3. Explainable AI & Reasoning**
    - **Human-Readable Explanations**: Generates specific reasons for each flagged transaction
    - **Context-Aware Analysis**: Considers transaction patterns, timing, amounts, and beneficiary relationships
    - **Actionable Insights**: Provides clear explanations of why transactions were flagged as suspicious

    **👥 4. Customer-Centric Analysis**
    - **Consolidated Customer Views**: Aggregates all flagged transactions per customer
    - **Risk Scoring**: Calculates z-scores and percentiles to prioritize high-risk customers
    - **Behavioral Stories**: Creates narrative summaries of customer transaction patterns
    - **Transaction Networks**: Visualizes customer connections and identifies potential laundering rings

    **🔗 5. Beneficiary Name Matching & Consolidation**
    - **Intelligent Name Matching**: Advanced fuzzy matching to identify similar beneficiary names
    - **Multi-Scenario Handling**: Detects hashed names, word order variations, partial names, and case differences
    - **Similarity Scoring**: Adjustable threshold-based matching with confidence scores
    - **Consolidated Analytics**: Aggregates transactions across all name variations
    - **Risk Assessment**: Evaluates consolidated beneficiary profiles for suspicious activity
    - **Pattern Detection**: Identifies potential name obfuscation and evasion techniques

    **🌐 6. Network Analysis & Graph Theory**
    - **Transaction Networks**: Builds directed graphs showing money flow between customers
    - **Cycle Detection**: Identifies circular transaction patterns (potential laundering rings)
    - **Hub Detection**: Finds customers with unusually high transaction volumes
    - **Interactive Visualization**: Compact graph display with node labels showing transaction counts

    **🤖 7. Graph RAG (Retrieval-Augmented Generation)**
    - **LLM-Powered Analysis**: Uses GPT-4o for intelligent transaction pattern analysis
    - **Risk Assessment**: Provides color-coded risk levels (RED=HIGH, GREEN=LOW, BLUE=MEDIUM)
    - **Concise Insights**: Delivers exactly 4 key bullet points with actionable recommendations
    - **Context-Aware**: Analyzes specific customer transaction histories for targeted insights

    **📈 8. Advanced Analytics & Visualization**
    - **Pie Charts**: Transaction count and sum distribution by transfer type
    - **Stacked Bar Charts**: Daily transaction amounts with hover details
    - **Network Graphs**: Interactive customer transaction networks with cycle highlighting
    - **Statistical Summaries**: Comprehensive transaction statistics and risk metrics

    **🔐 9. Privacy & Security Features**
    - **Data Hashing**: SHA-256 hashing of all customer identifiers
    - **Reversible Lookup**: Customer ID lookup tool for investigator access
    - **Session Management**: Secure data handling across tabs
    - **Audit Trail**: Complete transaction history preservation

    **📋 10. Investigator Tools**
    - **Customer ID Lookup**: Reverse hashing for flagged customer investigation
    - **Export Capabilities**: Download results in CSV format
    - **Multi-tab Interface**: Organized workflow across EDA, Detection, RAG, and Lookup
    - **Real-time Analysis**: Immediate results without batch processing delays

    **🎯 Key Benefits:**
    - **Comprehensive Coverage**: Multi-layered detection combining ML, graph theory, and LLM analysis
    - **Explainable Results**: Clear reasoning for every flagged transaction
    - **Privacy-First**: Customer data protection through hashing
    - **Actionable Insights**: Prioritized risk assessment with specific recommendations
    - **Scalable Architecture**: Handles large transaction datasets efficiently
    - **User-Friendly**: Intuitive interface for both technical and non-technical users
    - **Advanced Name Matching**: Intelligent beneficiary consolidation across name variations

    **🔬 Technical Stack:**
    - **Machine Learning**: Scikit-learn (Isolation Forest, StandardScaler)
    - **Graph Analysis**: NetworkX for transaction network modeling
    - **Visualization**: Plotly for interactive charts, Matplotlib for network graphs
    - **LLM Integration**: OpenAI GPT-4o for intelligent analysis
    - **Fuzzy Matching**: TheFuzz library for advanced name similarity detection
    - **Web Framework**: Streamlit for responsive web interface
    - **Data Processing**: Pandas for efficient data manipulation
    """
    )
