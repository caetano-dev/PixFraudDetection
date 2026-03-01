"""
Data loading utilities for the PixFraudDetection pipeline.

This module migrates ``load_data`` and ``get_bad_actors_up_to_date`` from the
legacy ``utils.py``.  All data-access concerns are isolated here so that the
rest of the pipeline can treat raw DataFrames as an opaque input.

Mathematical invariants preserved from the original implementation:
    - 1-to-N joint account ownership is handled via a Cartesian-expand
      relational merge, producing one edge row per (entity_src, entity_tgt) pair.
    - Conservation of Mass: transaction amounts are divided by
      (src_owner_count * tgt_owner_count) so that the total value flowing
      through any account is preserved after the expansion.
    - Time-aware bad-actor labelling: ``get_bad_actors_up_to_date`` returns
      only entities whose laundering activity is *known* up to a given date,
      preventing future-information leakage during temporal evaluation.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.config import (
    ACCOUNTS_FILE,
    DATA_PATH,
    LAUNDERING_TRANSACTIONS_FILE,
    NORMAL_TRANSACTIONS_FILE,
)


def load_data() -> tuple[pd.DataFrame, set]:
    """
    Load and preprocess all transaction data from the configured ``DATA_PATH``.

    Processing steps
    ----------------
    1. Load the three Parquet files (normal transactions, laundering
       transactions, accounts).
    2. Concatenate normal and laundering transactions into a single DataFrame.
    3. Map each account to its owning entity (or entities, for joint accounts)
       via relational merges, producing a row-expanded edge table.
    4. Drop rows whose ``from_account`` or ``to_account`` could not be mapped
       to any entity (unmapped / orphaned accounts).
    5. Enforce Conservation of Mass by fractionally dividing
       ``amount_sent_c`` and ``amount_received`` by the Cartesian product of
       the source and target account ownership degrees.
    6. Derive the global bad-actor set from every laundering transaction in
       the full dataset.

    Returns
    -------
    all_transactions : pd.DataFrame
        Merged, entity-resolved transaction table.  Key columns added by this
        function:

        * ``source_entity``  – entity ID of the sending party
        * ``target_entity``  – entity ID of the receiving party
        * ``amount_sent_c``  – conservation-of-mass-adjusted sent amount
        * ``amount_received``– conservation-of-mass-adjusted received amount
        * ``timestamp``      – parsed ``datetime64`` column

    bad_actors_global : set
        Entity IDs involved in *any* laundering transaction in the dataset.
        This is the global ground-truth label set and must **only** be used
        for final, post-hoc evaluation.  For time-sliced evaluation use
        :func:`get_bad_actors_up_to_date` instead.
    """
    normal_path = DATA_PATH / NORMAL_TRANSACTIONS_FILE
    laundering_path = DATA_PATH / LAUNDERING_TRANSACTIONS_FILE
    accounts_path = DATA_PATH / ACCOUNTS_FILE

    print(f"Loading data from {DATA_PATH}...")
    normal_transactions = pd.read_parquet(normal_path)
    laundering_transactions = pd.read_parquet(laundering_path)
    accounts = pd.read_parquet(accounts_path)

    # ------------------------------------------------------------------ #
    # 1. Concatenate all transactions                                      #
    # ------------------------------------------------------------------ #
    all_transactions = pd.concat(
        [normal_transactions, laundering_transactions],
        ignore_index=True,
    )

    all_transactions["timestamp"] = pd.to_datetime(all_transactions["timestamp"])

    # ------------------------------------------------------------------ #
    # 2. Compute per-account ownership degree                              #
    # ------------------------------------------------------------------ #
    # owner_count = number of entities that co-own an account (joint accounts
    # produce multiple rows in the accounts table for the same account number).
    owner_counts = accounts.groupby("Account Number")["Entity ID"].count().reset_index()
    owner_counts.columns = ["Account Number", "owner_count"]

    # Enriched account lookup: (Account Number, Entity ID, owner_count)
    acct_subset = accounts[["Account Number", "Entity ID"]].merge(
        owner_counts, on="Account Number"
    )

    # ------------------------------------------------------------------ #
    # 3. Map source accounts → entities                                    #
    # ------------------------------------------------------------------ #
    all_transactions = (
        all_transactions.merge(
            acct_subset,
            left_on="from_account",
            right_on="Account Number",
            how="left",
        )
        .rename(
            columns={"Entity ID": "source_entity", "owner_count": "src_owner_count"}
        )
        .drop(columns=["Account Number"])
    )

    # ------------------------------------------------------------------ #
    # 4. Map target accounts → entities                                    #
    # ------------------------------------------------------------------ #
    all_transactions = (
        all_transactions.merge(
            acct_subset,
            left_on="to_account",
            right_on="Account Number",
            how="left",
        )
        .rename(
            columns={"Entity ID": "target_entity", "owner_count": "tgt_owner_count"}
        )
        .drop(columns=["Account Number"])
    )

    # ------------------------------------------------------------------ #
    # 5. Drop unmapped accounts                                            #
    # ------------------------------------------------------------------ #
    initial_count = len(all_transactions)
    all_transactions = all_transactions.dropna(
        subset=["source_entity", "target_entity"]
    )
    dropped_count = initial_count - len(all_transactions)

    if dropped_count > 0:
        print(f"Dropped {dropped_count:,} transactions with unmapped accounts")

    # ------------------------------------------------------------------ #
    # 6. Conservation of Mass (Fractional Ownership)                       #
    # ------------------------------------------------------------------ #
    # When one account is jointly owned by N entities the single transaction
    # becomes N×M synthesised edges.  Dividing by (N * M) preserves the total
    # money-flow through every account.
    # AMLword paper: "each entity (person or company) owns one or more bank accounts directly and via subsidiaries"
    divisor = all_transactions["src_owner_count"] * all_transactions["tgt_owner_count"]
    all_transactions["amount_sent_c"] = all_transactions["amount_sent_c"] / divisor
    all_transactions["amount_received"] = all_transactions["amount_received"] / divisor

    # ------------------------------------------------------------------ #
    # 7. Derive global bad-actor set                                       #
    # ------------------------------------------------------------------ #
    laundering_txns = all_transactions[all_transactions["is_laundering"] == 1]
    bad_actors_global: set = set(laundering_txns["source_entity"]).union(
        set(laundering_txns["target_entity"])
    )

    print(f"Loaded {len(all_transactions):,} individual entity-to-entity edge records")
    print(
        f"Identified {len(bad_actors_global):,} bad actors globally "
        "(entities in laundering transactions)"
    )

    return all_transactions, bad_actors_global


def get_bad_actors_up_to_date(
    all_transactions: pd.DataFrame,
    current_date: datetime,
) -> set:
    """
    Return bad actors whose fraudulent activity is known up to *current_date*.

    This function is the correct label source for all temporal evaluation
    steps inside the sliding-window loop.  Using the global bad-actor set
    (returned by :func:`load_data`) inside the loop would leak future
    information — an entity flagged as a launderer on day T+30 would
    incorrectly inflate evaluation metrics for day T.

    Parameters
    ----------
    all_transactions : pd.DataFrame
        Full transaction DataFrame as returned by :func:`load_data`.
        Must contain ``timestamp``, ``is_laundering``, ``source_entity``,
        and ``target_entity`` columns.
    current_date : datetime
        Inclusive upper bound.  Only laundering transactions with
        ``timestamp <= current_date`` are considered.

    Returns
    -------
    set
        Entity IDs involved in laundering transactions up to and including
        *current_date*.
    """
    mask = all_transactions["timestamp"] <= current_date
    transactions_up_to_date = all_transactions.loc[mask]

    laundering_txns = transactions_up_to_date[
        transactions_up_to_date["is_laundering"] == 1
    ]
    bad_actors: set = set(laundering_txns["source_entity"]).union(
        set(laundering_txns["target_entity"])
    )

    return bad_actors
