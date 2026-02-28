"""
Abstract base class for the Strategy Pattern feature extractors.

Every concrete feature extractor in this package inherits from
:class:`FeatureExtractor` and must implement the single ``extract``
method.  The orchestrator in ``scripts/02_extract_features.py`` treats
all extractors through this uniform interface, making it trivial to
enable, disable, or swap algorithms without touching the pipeline loop.

Strategy Pattern roles
----------------------
* **Strategy interface** → :class:`FeatureExtractor` (this module)
* **Concrete strategies** → :mod:`src.features.centrality`,
  :mod:`src.features.community`, :mod:`src.features.stability`
* **Context** → ``scripts/02_extract_features.py``
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx


class FeatureExtractor(ABC):
    """
    Abstract strategy interface for graph feature extraction.

    All concrete extractors must override :meth:`extract`.  The method
    receives the fully-built directed graph for the current temporal
    window and returns a flat dictionary mapping each node ID to a
    dict of ``{feature_name: value}`` pairs.

    The contract guarantees that:

    * The return type is always ``dict[..., dict[str, float | int]]``.
    * On any recoverable algorithm failure the extractor returns an
      **empty dict** rather than raising, so the orchestrator can
      continue processing the remaining windows.
    * Extractors are **stateless by default**.  The one exception,
      :class:`~src.features.stability.RankStabilityTracker`, explicitly
      documents its statefulness.

    Parameters
    ----------
    (none at the base level — concrete subclasses may add their own)

    Examples
    --------
    Registering and running a collection of strategies::

        extractors: list[FeatureExtractor] = [
            PageRankVolumeExtractor(),
            PageRankFrequencyExtractor(),
            HITSExtractor(),
            LeidenCommunityExtractor(),
        ]

        for extractor in extractors:
            features = extractor.extract(G)
            # features: {node_id: {"pagerank": 0.002, ...}, ...}
    """

    @abstractmethod
    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float | int]]:
        """
        Extract features from *G* and return a per-node feature mapping.

        Parameters
        ----------
        G : nx.DiGraph
            The directed transaction graph for the current temporal window,
            as produced by :func:`src.graph.builder.build_daily_graph`.

        Returns
        -------
        dict
            A dictionary keyed by node ID.  Each value is itself a flat
            dictionary whose keys are feature names (``str``) and whose
            values are numeric (``float`` or ``int``).

            Returns an **empty dict** if *G* has no nodes or if the
            underlying algorithm fails to converge / raises a recoverable
            error.

        Notes
        -----
        Implementations must catch :exc:`networkx.NetworkXError` and
        :exc:`networkx.PowerIterationFailedConvergence` internally and
        return ``{}`` on failure rather than propagating the exception.
        """

    # ------------------------------------------------------------------
    # Convenience helpers available to all concrete subclasses
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Human-readable identifier for this extractor.

        Defaults to the class name.  Override in subclasses when a more
        descriptive label is needed (e.g. for logging or column prefixing).
        """
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{self.name}()"
