from abc import ABC, abstractmethod

import pandas as pd
from igraph import Graph, Vertex


class Graph2Text(ABC):
    @abstractmethod
    def __call__(self, graph: Graph) -> str:
        pass


class Graph2JSON(Graph2Text):
    def __call__(self, graph: Graph):
        """Converts a igraph.Graph object to JSON format."""
        nodes: list[dict] = []
        edges: list[dict] = []
        for node in graph.vs:
            _node = {"id": node.index, "label": node["name"]}
            _node.update(self.clean(node))
            nodes.append(_node)
        for edge in graph.es:
            edges.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge["Weight"],
                }
            )

        return {"nodes": nodes, "edges": edges}

    def clean(self, node: Vertex) -> dict:
        remove_keys = {
            "filePathReal",
        }

        filtered_attrs = {
            k: val for k, val in node.attributes().items() if k not in remove_keys
        }

        return filtered_attrs


class Graph2CSV(Graph2Text):
    def __call__(self, graph: Graph):
        """Converts a igraph.Graph object to JSON format."""
        nodes: list[dict] = []
        edges: list[dict] = []
        for node in graph.vs:
            _node = {"id": node.index, "label": node["name"]}
            _node.update(self.clean(node))
            nodes.append(_node)
        for edge in graph.es:
            edges.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge["Weight"],
                }
            )
        nodes: pd.DataFrame = pd.DataFrame(nodes)
        edges: pd.DataFrame = pd.DataFrame(edges)
        return {"nodes": nodes.to_csv(index=False), "edges": edges.to_csv(index=False)}

    def clean(self, node: Vertex) -> dict:
        remove_keys = {"filePathReal"}

        filtered_attrs = {
            k: val for k, val in node.attributes().items() if k not in remove_keys
        }

        return filtered_attrs
