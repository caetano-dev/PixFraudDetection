"""
Temporal window generator for the PixFraudDetection sliding-window pipeline.

This module abstracts the ``while current_date <= end_date:`` loop that lived
inside ``main.py`` into a clean, reusable iterator.  Consumers only need to
instantiate :class:`TemporalWindowGenerator` and iterate over it — the
bookkeeping for window boundaries, step advancement, and empty-window skipping
is fully encapsulated here.

Design notes
------------
* The generator is implemented as a class with ``__iter__`` / ``__next__`` so
  that it can be introspected (``len``, ``repr``) without consuming the
  iterator, which is useful for progress-bar initialisation in the
  orchestrator.
* Window boundaries mirror the original pipeline exactly:
      window_start = current_date - Timedelta(days=window_days)
      mask: timestamp > window_start AND timestamp <= current_date
  The half-open interval ``(window_start, current_date]`` means the window
  is *exclusive* of the start boundary and *inclusive* of the end boundary,
  matching the original ``main.py`` semantics precisely.
* Empty windows (weekends / public holidays with zero transactions) are
  silently skipped and the date is advanced by ``step_size`` as before.
* The first ``current_date`` is ``start_date + Timedelta(days=1)`` so that
  the very first window can grow from a single day up to ``window_days``
  days, replicating the original "warm-up" behaviour.
"""

from __future__ import annotations

from typing import Iterator, cast

import pandas as pd


class TemporalWindowGenerator:
    """
    Iterate over a transactions DataFrame day-by-day with a sliding window.

    Each iteration step yields ``(current_date, window_df)`` where
    ``window_df`` contains all transactions in the half-open interval
    ``(current_date - window_days, current_date]``.  Empty windows are
    skipped automatically.

    Parameters
    ----------
    transactions : pd.DataFrame
        Full transaction table as returned by
        :func:`src.data.loader.load_data`.  Must contain a ``timestamp``
        column of dtype ``datetime64``.
    window_days : int
        Width of the lookback window in calendar days.
    step_size : int
        Number of days to advance ``current_date`` on each iteration.
        Defaults to ``1``.

    Examples
    --------
    >>> gen = TemporalWindowGenerator(all_transactions, window_days=3)
    >>> for current_date, window_df in gen:
    ...     G = build_daily_graph(window_df)
    ...     # … extract features …
    """

    def __init__(
        self,
        transactions: pd.DataFrame,
        window_days: int,
        step_size: int = 1,
    ) -> None:
        if transactions.empty:
            raise ValueError(
                "TemporalWindowGenerator received an empty transactions DataFrame."
            )
        if "timestamp" not in transactions.columns:
            raise ValueError(
                "transactions DataFrame must contain a 'timestamp' column."
            )
        if window_days < 1:
            raise ValueError(f"window_days must be >= 1, got {window_days}.")
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}.")

        self._transactions = transactions
        self._window_days = window_days
        self._step_size = step_size

        # Derive the iteration bounds from the data — same logic as main.py.
        self._start_date: pd.Timestamp = cast(
            pd.Timestamp, transactions["timestamp"].min()
        )
        self._end_date: pd.Timestamp = cast(
            pd.Timestamp, transactions["timestamp"].max()
        )

        # The first current_date is start + 1 day so the very first window
        # can be a partial (growing) window of a single day, identical to
        # the original pipeline's warm-up behaviour.
        self._first_date: pd.Timestamp = cast(
            pd.Timestamp, self._start_date + pd.Timedelta(days=1)
        )

        # Pre-compute total number of steps so callers can use len() for
        # progress-bar initialisation without consuming the iterator.
        self._total_steps: int = (
            self._end_date - self._first_date
        ).days // self._step_size + 1

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def start_date(self) -> pd.Timestamp:
        """Earliest timestamp in the transactions DataFrame."""
        return self._start_date

    @property
    def end_date(self) -> pd.Timestamp:
        """Latest timestamp in the transactions DataFrame."""
        return self._end_date

    @property
    def total_steps(self) -> int:
        """
        Upper bound on the number of iterations (includes potentially empty
        windows that will be skipped at runtime).
        """
        return self._total_steps

    def __len__(self) -> int:
        """Return the upper bound on iteration steps (same as ``total_steps``)."""
        return self._total_steps

    def __repr__(self) -> str:
        return (
            f"TemporalWindowGenerator("
            f"start={self._first_date.date()}, "
            f"end={self._end_date.date()}, "
            f"window_days={self._window_days}, "
            f"step_size={self._step_size}, "
            f"total_steps={self._total_steps})"
        )

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
        """
        Yield ``(current_date, window_df)`` tuples for every non-empty window.

        The iteration variable ``current_date`` always represents the
        **right / inclusive** boundary of the window.  The left boundary is
        always ``current_date - Timedelta(days=window_days)`` (exclusive).

        Yields
        ------
        current_date : pd.Timestamp
            The right-inclusive boundary date of the current window.
        window_df : pd.DataFrame
            Subset of ``transactions`` whose ``timestamp`` falls in the
            half-open interval ``(current_date - window_days, current_date]``.
            Empty windows are skipped — this DataFrame is always non-empty.
        """
        current_date: pd.Timestamp = self._first_date
        step: pd.Timedelta = cast(pd.Timedelta, pd.Timedelta(days=self._step_size))
        window_lookback: pd.Timedelta = cast(
            pd.Timedelta, pd.Timedelta(days=self._window_days)
        )

        while current_date <= self._end_date:
            window_start = current_date - window_lookback

            # Replicate the original half-open mask:
            #   timestamp > window_start  AND  timestamp <= current_date
            mask = (self._transactions["timestamp"] > window_start) & (
                self._transactions["timestamp"] <= current_date
            )
            window_df = self._transactions.loc[mask].copy()

            # Skip empty windows (weekends / public holidays) exactly as
            # the original while-loop did, but do so silently inside the
            # iterator so callers never receive an empty DataFrame.
            if not window_df.empty:
                yield current_date, window_df

            current_date += step
