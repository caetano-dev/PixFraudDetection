import streamlit as st
import pandas as pd
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from streamlit_agraph import agraph, Node, Edge, Config
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="PIX Fraud Detection Dashboard",
    page_icon="🕵️",
    layout="wide",
)

# --- Environment and Data Loading ---
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

@st.cache_data
def load_data(filepath):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Ensure fraudTransactionCount exists, fill with 0 if not
        if 'fraudTransactionCount' not in df.columns:
            df['fraudTransactionCount'] = 0
        return df
    return None

@st.cache_data
def load_community_data():
    community_analysis = load_data("community_analysis.csv")
    lof_analysis = load_data("lof_analysis.csv")
    lof_community_summary = load_data("lof_community_summary.csv")
    return community_analysis, lof_analysis, lof_community_summary

@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

driver = get_neo4j_driver()
anomaly_data = load_data("anomaly_scores.csv")
community_analysis, lof_analysis, lof_community_summary = load_community_data()

# --- Neo4j Queries ---
class Neo4jConnection:
    def get_kpis(self):
        query = """
        MATCH (t:Transaction)
        WITH
            count(t) AS totalTransactions,
            sum(t.amount) AS totalAmount,
            sum(CASE WHEN t.fraudFlag STARTS WITH 'SMURFING' OR t.fraudFlag = 'CIRCULAR_PAYMENT' THEN t.amount ELSE 0 END) AS totalFraudAmount,
            sum(CASE WHEN t.fraudFlag STARTS WITH 'SMURFING' OR t.fraudFlag = 'CIRCULAR_PAYMENT' THEN 1 ELSE 0 END) AS totalFraudTransactions
        RETURN
            totalTransactions,
            totalAmount,
            totalFraudAmount,
            totalFraudTransactions
        """
        with driver.session() as session:
            result = session.run(query)
            return result.single()

    def get_fraud_counts_by_type(self):
        query = """
        MATCH (t:Transaction)
        WHERE t.fraudFlag STARTS WITH 'SMURFING' OR t.fraudFlag = 'CIRCULAR_PAYMENT'
        RETURN t.fraudFlag AS fraudType, count(t) AS count
        ORDER BY count DESC
        """
        with driver.session() as session:
            result = session.run(query)
            return pd.DataFrame([r.data() for r in result])

    def get_account_neighborhood(self, account_id, hop_count=1):
        query = """
        MATCH (a:Account {accountId: $account_id})-[r*1..%d]-(neighbor)
        UNWIND r as rel
        RETURN DISTINCT
            startNode(rel).accountId AS source,
            endNode(rel).accountId AS target,
            type(rel) as type
        """ % hop_count
        with driver.session() as session:
            result = session.run(query, account_id=account_id)
            return pd.DataFrame([r.data() for r in result])

# --- UI Rendering ---
st.title("🕵️ PIX Fraud Detection Dashboard")

if anomaly_data is None:
    st.error("`anomaly_scores.csv` not found. Please run the `feature_engineering.py` and `anomaly_detection.py` scripts first.")
    st.stop()

# Sort by anomaly score to show most suspicious accounts first
anomaly_data = anomaly_data.sort_values(by="anomaly_score", ascending=False)

# --- Main App Logic ---
conn = Neo4jConnection()
kpis = conn.get_kpis()

# Create tabs
tab1, tab2, tab3 = st.tabs(["📈 High-Level Metrics", "🚨 Alerts & Investigation", "🔍 Community Analysis"])

with tab1:
    st.header("Platform-Wide Metrics")
    if kpis:
        fraud_rate = (kpis['totalFraudTransactions'] / kpis['totalTransactions']) * 100 if kpis['totalTransactions'] > 0 else 0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{kpis['totalTransactions']:,}")
        col2.metric("Total Volume", f"R$ {kpis['totalAmount']:,.2f}")
        col3.metric("Fraudulent Transactions", f"{kpis['totalFraudTransactions']:,}")
        col4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    st.header("Fraud Breakdown by Type")
    fraud_counts_df = conn.get_fraud_counts_by_type()
    if not fraud_counts_df.empty:
        fig = px.bar(
            fraud_counts_df,
            x="fraudType",
            y="count",
            title="Count of Fraudulent Transactions by Type",
            labels={"fraudType": "Fraud Type", "count": "Number of Transactions"},
            color="fraudType",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fraudulent transactions found to display in the chart.")

with tab2:
    st.header("Anomalous Account Investigation")

    st.subheader("Top Anomalous Accounts")
    st.write("These accounts have been flagged by the Isolation Forest model as having the most unusual transaction patterns. Investigate their neighborhoods to uncover potential fraud rings.")
    
    # Display a searchable and sortable table of anomalous accounts
    st.dataframe(
        anomaly_data,
        use_container_width=True,
        column_config={
            "accountId": "Account ID",
            "anomaly_score": st.column_config.NumberColumn(
                "Anomaly Score",
                help="Lower scores are more anomalous.",
                format="%.4f",
            ),
        },
        hide_index=True,
    )

    st.subheader("Investigate Account Neighborhood")
    
    # Dropdown to select an account to investigate
    selected_account = st.selectbox(
        "Select an Account ID to Investigate:",
        options=anomaly_data["accountId"].unique(),
        help="Choose an account from the list above to visualize its connections."
    )

    if selected_account:
        st.write(f"Rendering graph for **{selected_account}**...")
        
        # Get neighborhood data
        neighborhood_df = conn.get_account_neighborhood(selected_account, hop_count=2)

        if not neighborhood_df.empty:
            nodes = []
            edges = []
            unique_accounts = set(neighborhood_df['source']).union(set(neighborhood_df['target']))

            for acc in unique_accounts:
                # Highlight the selected node
                node_color = "#FF4B4B" if acc == selected_account else "#ADD8E6"
                node_size = 25 if acc == selected_account else 15
                nodes.append(Node(id=acc, label=acc, size=node_size, color=node_color))

            for index, row in neighborhood_df.iterrows():
                edges.append(Edge(source=row['source'], target=row['target'], label=row['type']))

            # Configure graph visualization
            config = Config(width=1000,
                            height=800,
                            directed=True,
                            physics=True,
                            hierarchical=False,
                            )

            # Display the graph
            agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.warning(f"No transaction neighborhood found for account **{selected_account}** within 2 hops.")

with tab3:
    st.header("Community Analysis (Louvain + LOF)")
    
    if community_analysis is not None and not community_analysis.empty:
        st.subheader("Community Overview")
        st.write("Communities detected using the Louvain algorithm, sorted by fraud risk.")
        
        # Display community analysis table
        st.dataframe(
            community_analysis.head(20),
            use_container_width=True,
            column_config={
                "communityId": "Community ID",
                "communitySize": "Size",
                "avgRiskScore": st.column_config.NumberColumn(
                    "Avg Risk Score",
                    format="%.3f",
                ),
                "fraudRate": st.column_config.NumberColumn(
                    "Fraud Rate",
                    format="%.3f",
                ),
                "totalFraudTransactions": "Fraud Transactions",
                "totalAmount": st.column_config.NumberColumn(
                    "Total Amount",
                    format="R$ %.2f",
                ),
            },
            hide_index=True,
        )
        
        # Community fraud rate chart
        if 'fraudRate' in community_analysis.columns:
            fig_community = px.bar(
                community_analysis.head(10),
                x="communityId",
                y="fraudRate",
                title="Top 10 Communities by Fraud Rate",
                labels={"communityId": "Community ID", "fraudRate": "Fraud Rate"},
                color="fraudRate",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_community, use_container_width=True)
    else:
        st.info("Community analysis data not found. Run `python community_detection.py` first.")
    
    if lof_analysis is not None and not lof_analysis.empty:
        st.subheader("Local Outlier Factor (LOF) Results")
        st.write("Accounts that are outliers within their own communities.")
        
        # Show top LOF outliers
        lof_outliers = lof_analysis[lof_analysis['is_lof_outlier'] == 1].sort_values('lof_score').head(20)
        
        if not lof_outliers.empty:
            st.dataframe(
                lof_outliers[['accountId', 'community', 'lof_score', 'riskScore', 'fraudulentTransactions', 'totalTransactionAmount']],
                use_container_width=True,
                column_config={
                    "accountId": "Account ID",
                    "community": "Community",
                    "lof_score": st.column_config.NumberColumn(
                        "LOF Score",
                        help="Lower (more negative) scores are more anomalous.",
                        format="%.4f",
                    ),
                    "riskScore": st.column_config.NumberColumn(
                        "Risk Score",
                        format="%.3f",
                    ),
                    "fraudulentTransactions": "Fraud Transactions",
                    "totalTransactionAmount": st.column_config.NumberColumn(
                        "Total Amount",
                        format="R$ %.2f",
                    ),
                },
                hide_index=True,
            )
        else:
            st.info("No LOF outliers detected.")
        
        if lof_community_summary is not None and not lof_community_summary.empty:
            st.subheader("Community LOF Summary")
            st.dataframe(
                lof_community_summary.head(15),
                use_container_width=True,
                column_config={
                    "community": "Community ID",
                    "total_members": "Members",
                    "outlier_count": "Outliers",
                    "outlier_rate": st.column_config.NumberColumn(
                        "Outlier Rate",
                        format="%.2%",
                    ),
                    "avg_risk_score": st.column_config.NumberColumn(
                        "Avg Risk Score",
                        format="%.3f",
                    ),
                },
                hide_index=True,
            )
    else:
        st.info("LOF analysis data not found. Run `python local_outlier_factor.py` first.")
