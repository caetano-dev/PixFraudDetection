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
Longest laundering chain found: 7 days (filtered HI large)

Run clan_dataset to remove the laundering transactions we don't need.

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