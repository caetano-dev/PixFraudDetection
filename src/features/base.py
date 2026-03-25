from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float | int]]:
        """
        """

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{self.name}()"
