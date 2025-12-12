# PIX Fraud Detector

## To-do
- [X] remove transactions that come after number of days spanned inside of buildGraph.py (dataset dependent)
- [ ] sliding window for pagerank???

```
..                                  SMALL           MEDIUM           LARGE
..                                  HI     LI        HI      LI       HI       LI
.. Date Range HI + LI (2022)         Sep 1-10         Sep 1-16        Aug 1 - Nov 5
.. # of Days Spanned                 10     10        16      16       97       97
.. # of Bank Accounts               515K   705K     2077K   2028K    2116K    2064K
.. # of Transactions                  5M     7M       32M     31M      180M    176M
.. # of Laundering Transactions     5.1K   4.0K       35K     16K      223K    100K
.. Laundering Rate (1 per N Trans)  981   1942       905    1948       807     1750
..                                  SMALL           MEDIUM           LARGE
```

A Graph Neural Network (GNN) based Anti-Money Laundering (AML) detection system that constructs transaction graphs and identifies suspicious entities.

## Overview

This project builds a **directed weighted transaction graph** from financial transaction data, where:
- **Nodes** represent Entities (companies, individuals)
- **Edges** represent aggregated transaction flows between entities
- **Features** are engineered for both nodes and edges to enable GNN-based fraud detection

## Quick Start

### Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or: source venv/bin/activate.fish  # For fish shell
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Build the Graph

```bash
python buildGraph.py
```

## Graph Statistics

After processing the sample data:

| Metric | Value |
|--------|-------|
| Nodes (Entities) | 228,963 |
| Edges (Transaction flows) | 1,332,325 |
| Graph Density | 0.000025 |
| Entities with laundering label | 74,641 (32.60%) |
| Temporal Range | Aug 1, 2022 - Jan 12, 2023 |

## Feature Engineering

### Edge Attributes

| Feature | Description |
|---------|-------------|
| `weight` | Total transaction volume (sum of `amount_sent_c`) |
| `count` | Number of transactions between entity pair |

### Node Attributes

| Category | Feature | Description |
|----------|---------|-------------|
| **Inherent** | `label` | 1 if involved in any laundering transaction, 0 otherwise |
| **Structural** | `in_degree` | Number of incoming edges |
| | `out_degree` | Number of outgoing edges |
| | `in_strength` | Total money received (sum of incoming edge weights) |
| | `out_strength` | Total money sent (sum of outgoing edge weights) |
| | `avg_in_amount` | Average incoming transaction amount |
| | `avg_out_amount` | Average outgoing transaction amount |
| **Temporal** | `sent_amount_mean` | Mean of sent transaction amounts |
| | `sent_amount_max` | Maximum sent transaction amount |
| | `sent_amount_min` | Minimum sent transaction amount |
| | `sent_amount_std` | Standard deviation of sent amounts |
| | `received_amount_mean` | Mean of received transaction amounts |
| | `received_amount_max` | Maximum received transaction amount |
| | `received_amount_min` | Minimum received transaction amount |
| | `received_amount_std` | Standard deviation of received amounts |

## Data Schema

### Input Files (in `data/`)

#### 1. `1_filtered_normal_transactions.parquet`
Normal (legitimate) transactions.

| Column | Description |
|--------|-------------|
| `timestamp` | Transaction datetime |
| `from_bank` | Source bank ID |
| `from_account` | Source account ID |
| `to_bank` | Destination bank ID |
| `to_account` | Destination account ID |
| `amount_sent_c` | Transaction amount |
| `currency_sent` | Currency of transaction |
| `is_laundering` | Always 0 for this file |

#### 2. `2_filtered_laundering_transactions.parquet`
Fraudulent/laundering transactions (same schema as above, with `is_laundering=1`).

#### 3. `3_filtered_accounts.parquet`
Account to Entity mapping.

| Column | Description |
|--------|-------------|
| `Bank Name` | Name of the bank |
| `Bank ID` | Bank identifier |
| `Account Number` | Account identifier |
| `Entity ID` | Entity identifier (used as graph nodes) |
| `Entity Name` | Entity name/description |

## Sample Output

```
Node: 2AA02E7E570
  label: 1
  in_degree: 6
  out_degree: 8
  in_strength: 161,463,346
  out_strength: 131,818,682,426
  avg_in_amount: 26,910,557.67
  avg_out_amount: 16,477,335,303.25
  sent_amount_mean: 300,270,347.21
  sent_amount_std: 559,995,077.70
  ...

Edge: 2AA02E7E570 -> 2AA02EB7A00
  weight: 131,004,792,099
  count: 97
```

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `networkx` - Graph construction and analysis
- `pyarrow` - Parquet file support