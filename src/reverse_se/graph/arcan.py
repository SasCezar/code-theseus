from pathlib import Path

import igraph
import pandas as pd
from igraph import Graph
from pandas import DataFrame

from reverse_se.core.types import RepoContext


class ArcanGraph:
    def __init__(self, path: str, clean: bool = False):
        self.clean_edges: list[str] = [
            "isChildOf",
            "isImplementationOf",
            "nestedTo",
            "belongsTo",
            "implementedBy",
            "definedBy",
            "containerIsAfferentOf",
            "unitIsAfferentOf",
        ]
        self.clean: bool = clean
        self.graph: Graph = self.load(path)

    def load(self, path: str) -> Graph:
        graph: Graph = igraph.Graph.Read_GraphML(path)
        graph: Graph = self._clean_graph(graph) if self.clean else graph
        return graph

    def _clean_graph(self, graph: Graph) -> Graph:
        graph.es["weight"] = graph.es["Weight"]
        delete = [
            x.index for x in graph.vs if "$" in x["name"] and x["labelV"] != "container"
        ]
        graph.delete_vertices(delete)
        for edge_label in self.clean_edges:
            graph.es.select(labelE=edge_label).delete()

        graph.vs.select(_degree=0).delete()

        return graph


class ArcanCSV:
    def __init__(
        self,
        name: str,
        path: Path | str,
        clean: bool = True,
        package_only: bool = False,
        use_package_files: bool = True,
    ):
        self.name: str = name
        self.clean_edges_cols: list[str] = [
            "project",
            "versionId",
            "versionIndex",
            "versionDate",
            "InferSourceList",
            "Inferred",
            "LocationList",
            "SmellExtentByWeight",
            "edgeLabel",
            "from",
            "to",
            "edgeId",
        ]

        self.clean_nodes_cols: list[str] = [
            "project",
            "versionId",
            "versionIndex",
            "versionDate",
            "AbstractnessMetric",
            "ChangeHasOccurred",
            "InstabilityMetric",
            "NumberOfChildren",
            "TotalAmountOfChanges",
            "filePathReal",
            "simpleName",
            "vertexLabel",
            "constructType",
            "IsNested",
            "filePathRelative",
        ]

        self.clean: bool = clean
        self.package_only: bool = package_only
        self.edges: DataFrame = pd.read_csv(Path(path) / "component-dependencies.csv")
        self.nodes: DataFrame = pd.read_csv(Path(path) / "component-metrics.csv")
        self.membership: DataFrame = pd.read_csv(
            Path(path) / "component-membership.csv"
        )

        self.membership["files"] = self.membership.apply(
            lambda r: r["to"].replace(r["from"] + ".", "", 1), axis=1
        )

        self.package_files: DataFrame | None = (
            self.membership.groupby("from")["files"]
            .agg(list)
            .reset_index()
            .rename(columns={"from": "package"})
            if use_package_files
            else None
        )

        if self.package_only:
            self.nodes: DataFrame = self.nodes[self.nodes["vertexLabel"] == "container"]
            remaining_nodes: set[str] = set(self.nodes["vertexId"])
            self.edges: DataFrame = self.edges[
                self.edges["fromId"].isin(remaining_nodes)
                & self.edges["toId"].isin(remaining_nodes)
            ]
        if self.clean:
            self.edges: DataFrame = self._clean_edges(self.edges)
            self.nodes: DataFrame = self._clean_nodes(self.nodes)

        self.graph: tuple[DataFrame, DataFrame, DataFrame | None] = (
            self.nodes,
            self.edges,
            self.package_files,
        )

    @property
    def repository(self):
        return RepoContext(
            url="",
            language="",
            name=self.name,
            nodes=self.nodes,
            edges=self.edges,
            package_files=self.package_files,
        )

    def _clean_edges(self, edges: DataFrame) -> DataFrame:
        return edges.drop(columns=self.clean_edges_cols, errors="ignore")

    def _clean_nodes(self, nodes: DataFrame) -> DataFrame:
        return nodes.drop(columns=self.clean_nodes_cols, errors="ignore")
