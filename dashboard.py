import streamlit as st
import pandas as pd
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- Page Configuration ---
st.set_page_config(
    page_title="PIX Fraud Detection Dashboard",
    page_icon="🕵️",
    layout="wide",
)

# --- Data Loading ---
@st.cache_data
def load_data(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

anomaly_data = load_data("anomaly_scores.csv")

# --- Neo4j Connection ---
class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def get_account_details(self, account_id):
        query = """
        MATCH (a:Account {accountId: $account_id})
        RETURN a.accountId as accountId, a.risk_score as riskScore, a.state as state, a.is_verified as isVerified
        """
        with self._driver.session() as session:
            result = session.run(query, account_id=account_id)
            return result.single()

# --- UI Components ---
st.title("🕵️ PIX Fraud Detection Dashboard")

if anomaly_data is None:
    st.error("`anomaly_scores.csv` not found. Please run the `feature_engineering.py` and `anomaly_detection.py` scripts first.")
else:
    st.header("Anomaly Detection Results")
    st.write("This table shows accounts flagged as potential outliers by the Isolation Forest model. Lower anomaly scores indicate a higher likelihood of being an outlier.")

    # Display top anomalous accounts
    top_anomalies = anomaly_data[anomaly_data['is_outlier'] == 1].sort_values('anomaly_score')
    st.dataframe(top_anomalies)

    st.header("Account Investigation")
    
    # Search for an account
    account_id_to_search = st.text_input("Enter Account ID to investigate:", placeholder="e.g., 123.456.789-00")

    if account_id_to_search:
        account_details_df = anomaly_data[anomaly_data['accountId'] == account_id_to_search]

        if not account_details_df.empty:
            st.subheader(f"Details for Account: {account_id_to_search}")
            
            # Connect to Neo4j to get latest details
            conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
            neo4j_details = conn.get_account_details(account_id_to_search)
            conn.close()

            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Anomaly Score", f"{account_details_df.iloc[0]['anomaly_score']:.4f}")
                st.metric("Calculated Risk Score", f"{account_details_df.iloc[0]['riskScore']:.2f}")
                st.metric("Total Transaction Amount", f"R$ {account_details_df.iloc[0]['totalAmount']:,.2f}")

            with col2:
                st.metric("In-Degree", account_details_df.iloc[0]['inDegree'])
                st.metric("Out-Degree", account_details_df.iloc[0]['outDegree'])
                st.metric("Total Transactions", account_details_df.iloc[0]['transactionCount'])

            if neo4j_details:
                 st.info(f"""
                    - **State:** {neo4j_details['state']}
                    - **Verified Account:** {'Yes' if neo4j_details['isVerified'] else 'No'}
                """)
            
            st.dataframe(account_details_df)

        else:
            st.warning("Account ID not found in the anomaly results.")
