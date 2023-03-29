import dataclasses
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import cdt
import dataclass_cli
import networkx as nx
import numpy as np
import pandas as pd
from cdt.causality.graph import CGNN, GES, GIES, PC, SAM, CCDr
from cdt.causality.graph.bnlearn import GS, IAMB, MMPC, Fast_IAMB, Inter_IAMB
from src.features.preprocessing.evdef import EventDefinitionMap
from src.features.preprocessing.ts_transformation import (
    TimeSeriesTransformer, TimeSeriesTransformerConfig)
from tqdm import tqdm

from .base import Preprocessor
from .drain import Drain, DrainParameters


@dataclass_cli.add
@dataclasses.dataclass
class HDFSPreprocessorConfig:
    aggregated_log_file: Path = Path("data/logs_HDFS.csv")
    final_log_file: Path = Path("data/hdfs.pkl")
    relevant_aggregated_log_columns: List[str] = dataclasses.field(
        default_factory=lambda: [
            "Level",
            "Component",
            "Pid",
        ],
    )
    aggregate_per_max_number: int = -1
    aggregate_per_time_frequency: str = ""
    log_datetime_column_name: str = "timestamp"
    log_datetime_format: str = "%Y-%m-%d %H:%M:%S"
    log_payload_column_name: str = "Payload"
    label_column_name: str = "Label"
    include_label_in_knowledge: bool = False
    use_log_hierarchy: bool = False
    fine_drain_log_depth: int = 10
    fine_drain_log_st: float = 0.75
    coarse_drain_log_depth: int = 4
    coarse_drain_log_st: float = 0.2
    drain_log_depths: List[int] = dataclasses.field(default_factory=lambda: [],)
    drain_log_sts: List[float] = dataclasses.field(default_factory=lambda: [],)
    add_log_clusters: bool = True
    min_causality: float = 0.0
    log_only_causality: bool = False
    relevant_log_column: str = "fine_log_cluster_template"
    log_template_file: Path = Path("data/attention_log_templates.csv")
    r_path = '/usr/bin/Rscript'
    causal_algorithm_alpha: float = 0.05


class HDFSLogsPreprocessor(Preprocessor):
    sequence_column_name: str = "all_events"

    def __init__(self, config: HDFSPreprocessorConfig):
        self.config = config
        self.relevant_columns = set(
            [x for x in self.config.relevant_aggregated_log_columns]
        )
        self.relevant_columns.add("fine_log_cluster_template")
        self.relevant_columns.add("coarse_log_cluster_template")

        if self.config.include_label_in_knowledge:
            self.relevant_columns.add(self.config.label_column_name)

        for i in range(len(self.config.drain_log_depths)):
            self.relevant_columns.add(str(i) + "_log_cluster_template")

    def load_data(self) -> pd.DataFrame:
        log_only_data = self._load_log_only_data()
        log_only_data["grouper"] = 1
        return self._aggregate_per(log_only_data, aggregation_column="grouper")

    def _load_log_only_data(self) -> pd.DataFrame:
        log_df = self._read_log_df()
        for column in [x for x in log_df.columns if "log_cluster_template" in x]:
            log_df[column] = (
                log_df[column]
                .fillna("")
                .astype(str)
                .replace(np.nan, "", regex=True)
                .apply(lambda x: x if len(x) > 0 else "___empty___")
            )
        return log_df

    def _aggregate_per(
        self, merged_df: pd.DataFrame, aggregation_column: str = "parent_trace_id"
    ) -> pd.DataFrame:
        logging.debug("Aggregating HDFS data per %s", aggregation_column)
        for column in self.relevant_columns:
            merged_df[column] = merged_df[column].apply(
                lambda x: column + "#" + x.lower() if len(x) > 0 else ""
            )

        merged_df["all_events"] = merged_df[list(self.relevant_columns)].values.tolist()
        merged_df["attributes"] = merged_df[
            [x for x in self.relevant_columns if not "log_cluster_template" in x]
        ].values.tolist()
        for log_template_column in [
            x for x in self.relevant_columns if "log_cluster_template" in x
        ]:
            merged_df[log_template_column] = merged_df[log_template_column].apply(
                lambda x: [x]
            )
        events_per_trace = (
            merged_df.sort_values(by="timestamp")
            .groupby(aggregation_column)
            .agg(
                {
                    column_name: lambda x: list(x)
                    for column_name in ["all_events", "attributes",]
                    + [x for x in self.relevant_columns if "log_cluster_template" in x]
                }
            )
            .reset_index()
        )
        events_per_trace["num_logs"] = events_per_trace[
            self.config.relevant_log_column
        ].apply(lambda x: len([loglist for loglist in x if len(loglist[0]) > 0]))
        events_per_trace["num_events"] = events_per_trace[
            self.config.relevant_log_column
        ].apply(lambda x: len(x))
        return events_per_trace[
            ["num_logs", "num_events", "all_events", "attributes",]
            + [x for x in self.relevant_columns if "log_cluster_template" in x]
        ]

    def _read_log_df(self) -> pd.DataFrame:
        df = (
            pd.read_csv(self.config.aggregated_log_file)
            .fillna("")
            .astype(str)
            .replace(np.nan, "", regex=True)
        )

        rel_df = df[
            self.config.relevant_aggregated_log_columns
            + [self.config.log_datetime_column_name]
            + [self.config.log_payload_column_name]
        ]
        rel_df = self._add_log_drain_clusters(rel_df)
        if self.config.log_template_file.exists():
            rel_df = self._add_precalculated_log_templates(rel_df)
        rel_df["timestamp"] = pd.to_datetime(
            rel_df[self.config.log_datetime_column_name],
            format=self.config.log_datetime_format
        )
        return rel_df

    def _add_log_drain_clusters_prefix(
        self, log_df: pd.DataFrame, depth: int, st: float, prefix: str
    ) -> pd.DataFrame:
        all_logs_df = pd.DataFrame(
            log_df[self.config.log_payload_column_name].dropna().drop_duplicates()
        )
        drain = Drain(
            DrainParameters(
                depth=depth,
                st=st,
                rex=[
                    ("(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)", ""),
                    ("[^a-zA-Z\d\s:]", ""),
                ],
            ),
            data_df=all_logs_df,
            data_df_column_name=self.config.log_payload_column_name,
        )
        drain_result_df = drain.load_data().drop_duplicates().set_index("log_idx")
        log_result_df = (
            pd.merge(
                log_df,
                pd.merge(
                    all_logs_df,
                    drain_result_df,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
                .drop_duplicates()
                .reset_index(drop=True),
                on=self.config.log_payload_column_name,
                how="left",
            )
            .rename(
                columns={
                    "cluster_template": prefix + "log_cluster_template",
                    "cluster_path": prefix + "log_cluster_path",
                }
            )
            .drop(columns=["cluster_id"])
        )
        log_result_df[prefix + "log_cluster_template"] = (
            log_result_df[prefix + "log_cluster_template"]
            .fillna("")
            .astype(str)
            .replace(np.nan, "", regex=True)
        )
        return log_result_df

    def _add_precalculated_log_templates(self, log_df: pd.DataFrame) -> pd.DataFrame:
        precalculated_templates_df = pd.read_csv(self.config.log_template_file)
        if not "Payload" in precalculated_templates_df.columns:
            logging.error("Invalid log template file - does not contain Payload column!")
            return log_df
        self.relevant_columns.update(
            [x for x in precalculated_templates_df.columns if "log_cluster_template" in x]
        )
        return pd.merge(log_df, precalculated_templates_df, on="Payload", how="left")

    def _add_log_drain_clusters(self, log_df: pd.DataFrame) -> pd.DataFrame:
        log_result_df = self._add_log_drain_clusters_prefix(
            log_df=log_df,
            depth=self.config.fine_drain_log_depth,
            st=self.config.fine_drain_log_st,
            prefix="fine_",
        )
        log_result_df = self._add_log_drain_clusters_prefix(
            log_df=log_result_df,
            depth=self.config.coarse_drain_log_depth,
            st=self.config.coarse_drain_log_st,
            prefix="coarse_",
        )
        for i in range(len(self.config.drain_log_depths)):
            log_result_df = self._add_log_drain_clusters_prefix(
                log_df=log_result_df,
                depth=self.config.drain_log_depths[i],
                st=self.config.drain_log_sts[i],
                prefix=str(i) + "_",
            )
        return log_result_df

class HDFSLogsDescriptionPreprocessor(Preprocessor):
    def __init__(
        self, config: HDFSPreprocessorConfig,
    ):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        preprocessor = HDFSLogsPreprocessor(self.config)
        hdfs_df = preprocessor._load_log_only_data()
        return self._load_column_descriptions(hdfs_df, preprocessor.relevant_columns)

    def _load_column_descriptions(
        self, hdfs_df: pd.DataFrame, relevant_columns: Set[str]
    ) -> pd.DataFrame:
        column_descriptions = self._get_column_descriptions()
        description_records = []
        for column in relevant_columns:
            values = set(
                hdfs_df[column].dropna().astype(str).replace(np.nan, "", regex=True)
            )
            values = set([str(x).lower() for x in values if len(str(x)) > 0])
            for value in tqdm(values, desc="Loading descriptions for column " + column):
                description = " ".join(re.split("[,._\-\*]+", value))

                if column in column_descriptions:
                    description = column_descriptions[column] + " " + description

                description_records.append(
                    {"label": column + "#" + value, "description": description,},
                )

        return (
            pd.DataFrame.from_records(description_records)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _get_column_descriptions(self) -> Dict[str, str]:
        return {
            "Label": "Label",
            "Level": "Log level",
            "Code1": "Code 1",
            "Code2": "Code 2",
            "Component1": "Component 1",
            "Component2": "Component 2",
        }

class HDFSLogsHierarchyPreprocessor(Preprocessor):
    def __init__(
        self, config: HDFSPreprocessorConfig,
    ):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        if self.config.use_log_hierarchy:
            return self._load_log_only_hierarchy()
        else:
            return self._load_attribute_only_hierarchy()

    def _load_log_only_hierarchy(self) -> pd.DataFrame:
        preprocessor = HDFSLogsPreprocessor(self.config)
        hdfs_df = preprocessor._load_log_only_data()

        relevant_log_columns = set(
            [x for x in preprocessor.relevant_columns if "log_cluster_template" in x]
            + ["coarse_log_cluster_path"]
        )
        attribute_hierarchy = self._load_attribute_hierarchy(
            hdfs_df, set(["coarse_log_cluster_path"])
        )
        return (
            pd.concat(
            [attribute_hierarchy, self._load_log_hierarchy(hdfs_df, relevant_log_columns)], 
            ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _load_attribute_only_hierarchy(self) -> pd.DataFrame:
        preprocessor = HDFSLogsPreprocessor(self.config)
        hdfs_df = preprocessor._load_log_only_data()
        relevant_columns = set(
            [
                x
                for x in preprocessor.relevant_columns
                if "log_cluster_template" not in x
            ]
        )
        attribute_hierarchy = self._load_attribute_hierarchy(
            hdfs_df, relevant_columns
        )
        return (
            pd.concat([attribute_hierarchy, self._load_log_hierarchy(hdfs_df, relevant_columns)], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _load_log_hierarchy(
        self, hdfs_df: pd.DataFrame, relevant_columns: Set[str]
    ) -> pd.DataFrame:
        hierarchy_records = []
        for _, row in tqdm(
            hdfs_df.iterrows(),
            desc="Adding HDFS log hierarchy",
            total=len(hdfs_df),
        ):
            log_template = str(row[self.config.relevant_log_column]).lower()
            for column in relevant_columns:
                row_value = (
                    column + "#" + str(row[column]).lower()
                    if len(str(row[column])) > 0
                    else ""
                )
                if len(row_value) == 0:
                    continue

                hierarchy_records.append(
                    {
                        "parent_id": row_value,
                        "parent_name": row_value.split("#")[1],
                        "child_id": self.config.relevant_log_column + "#" + log_template,
                        "child_name": log_template,
                    },
                )
        return (
            pd.DataFrame.from_records(hierarchy_records)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _load_attribute_hierarchy(
        self, hdfs_df: pd.DataFrame, relevant_columns: Set[str]
    ) -> pd.DataFrame:
        hierarchy_df = pd.DataFrame(
            columns=["parent_id", "child_id", "parent_name", "child_name"]
        )
        for column in relevant_columns:
            hierarchy_df = pd.concat(
                    [hierarchy_df,
                     pd.DataFrame([{
                        "parent_id": "root",
                        "parent_name": "root",
                        "child_id": column,
                        "child_name": column,
                        }])
                    ], 
                    ignore_index=True)
            values = set(
                [
                    str(x).lower()
                    for x in hdfs_df[column]
                    .dropna()
                    .astype(str)
                    .replace(np.nan, "", regex=True)
                    if len(str(x)) > 0 and str(x).lower() != "nan"
                ]
            )
            for value in tqdm(values, desc="Loading hierarchy for column " + column):
                hierarchy_elements = [column]
                if "cluster" in column:
                    hierarchy_elements = hierarchy_elements + value.split()
                else:
                    hierarchy_elements = hierarchy_elements + re.split(
                        "[,._\-\*]+", value
                    )
                    hierarchy_elements = [
                        x.strip() for x in hierarchy_elements if len(x.strip()) > 0
                    ]
                if hierarchy_elements[len(hierarchy_elements) - 1] == value:
                    hierarchy_elements = hierarchy_elements[
                        : len(hierarchy_elements) - 1
                    ]

                hierarchy = []
                for i in range(1, len(hierarchy_elements) + 1):
                    hierarchy.append("->".join(hierarchy_elements[0:i]))
                hierarchy.append(column + "#" + value)

                parent_id = column
                parent_name = column
                for i in range(len(hierarchy)):
                    child_id = hierarchy[i]
                    child_name = child_id.split("->")[-1]
                    if not parent_id == child_id:
                        hierarchy_df = pd.concat(
                            [
                                hierarchy_df,
                                pd.DataFrame([
                                    {
                                        "parent_id": parent_id,
                                        "parent_name": parent_name,
                                        "child_id": child_id,
                                        "child_name": child_name,
                                    }
                                ])
                            ], ignore_index=True)
                    parent_id = child_id
                    parent_name = child_name

        return hierarchy_df[["parent_id", "child_id", "parent_name", "child_name"]]

class HDFSLogsCausalityPreprocessor(Preprocessor):
    def __init__(
        self, config: HDFSPreprocessorConfig,
    ):
        self.config = config
        self.algorithms = {
            'constraint': lambda df: PC(CItest = 'binary').predict(df),
            'score': lambda df: GES().predict(df),
            'CCDr': lambda df: CCDr().predict(df),
            'CGNN': lambda df: CGNN().predict(df), # Runs out of memory
            'SAM': lambda df: SAM(njobs=1, nruns=10).predict(df), # Runs of out memory
            'GIES': lambda df: GIES().predict(df),
            'GS-mi': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='mi').predict(df),
            'GS-mi-adf': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='mi-adf').predict(df),
            'GS-mc-mi': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='mc-mi').predict(df),
            'GS-smc-mi': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='smc-mi').predict(df),
            'GS-sp-mi': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='sp-mi').predict(df),
            'GS-mi-sh': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='mi-sh').predict(df),
            'GS-x2': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='x2').predict(df),
            'GS-x2-adf': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='x2-adf').predict(df),
            'GS-mc-x2': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='mc-x2').predict(df),
            'GS-smc-x2': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='smc-x2').predict(df),
            'GS-sp-x2': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='sp-x2').predict(df),
            'GS-jt': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='jt').predict(df),
            'GS-mc-jt': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='mc-jt').predict(df),
            'GS-smc-jt': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='smc-jt').predict(df),
            'GS-cor': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='cor').predict(df),
            'GS-mc-cor': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='mc-cor').predict(df),
            'GS-smc-cor': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='smc-cor').predict(df),
            'GS-zf': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='zf').predict(df),
            'GS-mc-zf': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='mc-zf').predict(df),
            'GS-smc-zf': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='smc-zf').predict(df),
            'GS-mi-g': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='mi-g').predict(df),
            'GS-mc-mi-g': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='mc-mi-g').predict(df),
            'GS-smc-mi-g': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='smc-mi-g').predict(df),
            'GS-mi-g-sh': lambda df: GS(alpha=self.config.causal_algorithm_alpha, score='mi-g-sh').predict(df),
            'IAMB-mi': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='mi').predict(df),
            'IAMB-mi-adf': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-adf').predict(df),
            'IAMB-mc-mi': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-mi').predict(df),
            'IAMB-smc-mi': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-mi').predict(df),
            'IAMB-sp-mi': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='sp-mi').predict(df),
            'IAMB-mi-sh': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-sh').predict(df),
            'IAMB-x2': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='x2').predict(df),
            'IAMB-x2-adf': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='x2-adf').predict(df),
            'IAMB-mc-x2': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-x2').predict(df),
            'IAMB-smc-x2': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-x2').predict(df),
            'IAMB-sp-x2': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='sp-x2').predict(df),
            'IAMB-jt': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='jt').predict(df),
            'IAMB-mc-jt': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-jt').predict(df),
            'IAMB-smc-jt': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-jt').predict(df),
            'IAMB-cor': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='cor').predict(df),
            'IAMB-mc-cor': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-cor').predict(df),
            'IAMB-smc-cor': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-cor').predict(df),
            'IAMB-zf': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='zf').predict(df),
            'IAMB-mc-zf': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-zf').predict(df),
            'IAMB-smc-zf': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-zf').predict(df),
            'IAMB-mi-g': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-g').predict(df),
            'IAMB-mc-mi-g': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-mi-g').predict(df),
            'IAMB-smc-mi-g': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-mi-g').predict(df),
            'IAMB-mi-g-sh': lambda df: IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-g-sh').predict(df),
            'Fast-IAMB-mi': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='mi').predict(df),
            'Fast-IAMB-mi-adf': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-adf').predict(df),
            'Fast-IAMB-mc-mi': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-mi').predict(df),
            'Fast-IAMB-smc-mi': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-mi').predict(df),
            'Fast-IAMB-sp-mi': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='sp-mi').predict(df),
            'Fast-IAMB-mi-sh': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-sh').predict(df),
            'Fast-IAMB-x2': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='x2').predict(df),
            'Fast-IAMB-x2-adf': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='x2-adf').predict(df),
            'Fast-IAMB-mc-x2': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-x2').predict(df),
            'Fast-IAMB-smc-x2': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-x2').predict(df),
            'Fast-IAMB-sp-x2': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='sp-x2').predict(df),
            'Fast-IAMB-jt': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='jt').predict(df),
            'Fast-IAMB-mc-jt': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-jt').predict(df),
            'Fast-IAMB-smc-jt': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-jt').predict(df),
            'Fast-IAMB-cor': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='cor').predict(df),
            'Fast-IAMB-mc-cor': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-cor').predict(df),
            'Fast-IAMB-smc-cor': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-cor').predict(df),
            'Fast-IAMB-zf': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='zf').predict(df),
            'Fast-IAMB-mc-zf': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-zf').predict(df),
            'Fast-IAMB-smc-zf': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-zf').predict(df),
            'Fast-IAMB-mi-g': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-g').predict(df),
            'Fast-IAMB-mc-mi-g': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-mi-g').predict(df),
            'Fast-IAMB-smc-mi-g': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-mi-g').predict(df),
            'Fast-IAMB-mi-g-sh': lambda df: Fast_IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-g-sh').predict(df),
            'Inter-IAMB-mi': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='mi').predict(df),
            'Inter-IAMB-mi-adf': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-adf').predict(df),
            'Inter-IAMB-mc-mi': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-mi').predict(df),
            'Inter-IAMB-smc-mi': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-mi').predict(df),
            'Inter-IAMB-sp-mi': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='sp-mi').predict(df),
            'Inter-IAMB-mi-sh': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-sh').predict(df),
            'Inter-IAMB-x2': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='x2').predict(df),
            'Inter-IAMB-x2-adf': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='x2-adf').predict(df),
            'Inter-IAMB-mc-x2': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-x2').predict(df),
            'Inter-IAMB-smc-x2': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-x2').predict(df),
            'Inter-IAMB-sp-x2': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='sp-x2').predict(df),
            'Inter-IAMB-jt': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='jt').predict(df),
            'Inter-IAMB-mc-jt': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-jt').predict(df),
            'Inter-IAMB-smc-jt': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-jt').predict(df),
            'Inter-IAMB-cor': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='cor').predict(df),
            'Inter-IAMB-mc-cor': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-cor').predict(df),
            'Inter-IAMB-smc-cor': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-cor').predict(df),
            'Inter-IAMB-zf': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='zf').predict(df),
            'Inter-IAMB-mc-zf': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-zf').predict(df),
            'Inter-IAMB-smc-zf': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-zf').predict(df),
            'Inter-IAMB-mi-g': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-g').predict(df),
            'Inter-IAMB-mc-mi-g': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='mc-mi-g').predict(df),
            'Inter-IAMB-smc-mi-g': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='smc-mi-g').predict(df),
            'Inter-IAMB-mi-g-sh': lambda df: Inter_IAMB(alpha=self.config.causal_algorithm_alpha, score='mi-g-sh').predict(df),
            'MMPC-mi': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='mi').predict(df),
            'MMPC-mi-adf': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='mi-adf').predict(df),
            'MMPC-mc-mi': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='mc-mi').predict(df),
            'MMPC-smc-mi': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='smc-mi').predict(df),
            'MMPC-sp-mi': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='sp-mi').predict(df),
            'MMPC-mi-sh': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='mi-sh').predict(df),
            'MMPC-x2': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='x2').predict(df),
            'MMPC-x2-adf': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='x2-adf').predict(df),
            'MMPC-mc-x2': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='mc-x2').predict(df),
            'MMPC-smc-x2': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='smc-x2').predict(df),
            'MMPC-sp-x2': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='sp-x2').predict(df),
            'MMPC-jt': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='jt').predict(df),
            'MMPC-mc-jt': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='mc-jt').predict(df),
            'MMPC-smc-jt': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='smc-jt').predict(df),
            'MMPC-cor': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='cor').predict(df),
            'MMPC-mc-cor': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='mc-cor').predict(df),
            'MMPC-smc-cor': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='smc-cor').predict(df),
            'MMPC-zf': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='zf').predict(df),
            'MMPC-mc-zf': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='mc-zf').predict(df),
            'MMPC-smc-zf': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='smc-zf').predict(df),
            'MMPC-mi-g': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='mi-g').predict(df),
            'MMPC-mc-mi-g': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='mc-mi-g').predict(df),
            'MMPC-smc-mi-g': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='smc-mi-g').predict(df),
            'MMPC-mi-g-sh': lambda df: MMPC(alpha=self.config.causal_algorithm_alpha, score='mi-g-sh').predict(df),
        }
        cdt.SETTINGS.rpath = config.r_path
        
    def load_data(self, algorithm = "heuristic") -> pd.DataFrame:
        preprocessor = HDFSLogsPreprocessor(self.config)
        hdfs_df = preprocessor._load_log_only_data().fillna("")
        
        relevant_columns = set(
            [
                x
                for x in preprocessor.relevant_columns
                if not self.config.log_only_causality or "log" in x
            ]
        )

        causality_records = []
        if algorithm == "heuristic":
            counted_causality = self._generate_counted_causality(
                hdfs_df, relevant_columns
            )

            for from_value, to_values in tqdm(
                counted_causality.items(),
                desc="Generating causality df from counted causality",
            ):
                total_to_counts = len(to_values)
                to_values_counter: Dict[str, int] = Counter(to_values)
                for to_value, to_count in to_values_counter.items():
                    if to_count / total_to_counts > self.config.min_causality:
                        causality_records.append(
                            {
                                "parent_id": from_value,
                                "parent_name": from_value.split("#")[1],
                                "child_id": to_value,
                                "child_name": to_value.split("#")[1],
                            },
                        )
        else:
            relevant_columns.add(self.config.log_datetime_column_name) # Heuristic doesn't use the timestamps
            transformer = TimeSeriesTransformer(TimeSeriesTransformerConfig(timestamp_column=self.config.log_datetime_column_name))
            transformed_df, evmap = transformer.transform_time_series_to_events(hdfs_df, relevant_columns)
            causality_records = self._generate_causality_records(transformed_df, evmap, algorithm)

        return (
            pd.DataFrame.from_records(causality_records)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _generate_counted_causality(
        self, df: pd.DataFrame, relevant_columns: Set[str]
    ) -> Dict[str, List[str]]:
        causality: Dict[str, List[str]] = {}
        previous_row = None
        for _, row in tqdm(
            df.iterrows(),
            desc="Generating counted causality for HDFS log data",
            total=len(df),
        ):
            if previous_row is None:
                previous_row = row
                continue
            for previous_column in relevant_columns:
                previous_column_value = (
                    previous_column + "#" + str(previous_row[previous_column]).lower()
                    if len(str(previous_row[previous_column])) > 0
                    else ""
                )
                if len(previous_column_value) < 1:
                    continue
                if previous_column_value not in causality:
                    causality[previous_column_value] = []
                for current_column in relevant_columns:
                    current_column_value = (
                        current_column + "#" + str(row[current_column]).lower()
                        if len(str(row[current_column])) > 0
                        else ""
                    )
                    if len(current_column_value) < 1:
                        continue
                    if current_column_value not in causality[previous_column_value]:
                        causality[previous_column_value].append(current_column_value)
                    else:
                        causality[previous_column_value].append(current_column_value)
            previous_row = row
        return causality
    
    def _generate_causality_records(self, transformed_df: pd.DataFrame, evmap: EventDefinitionMap, algorithm: str):
        if algorithm not in self.algorithms:
            logging.fatal(
                f"Causal knowledge algorithm {algorithm} is not available"
            )
            raise AlgorithmChoiceError(
                message=f"Causal knowledge algorithm {algorithm} is not available"
            )
        print(f'Generating causality with {algorithm} algorithm...')
        output: nx.Graph = self.algorithms[algorithm](transformed_df)
        causality_records = []
        for edge in list(output.edges()):
            from_eid, to_eid = edge[0], edge[1]
            from_evdef = evmap.get_evdef(from_eid)
            to_evdef = evmap.get_evdef(to_eid)
            causality_records.append(
                {
                    "parent_id": from_evdef.type + '#' + str(from_evdef.value),
                    "parent_name": from_evdef.value,
                    "child_id": to_evdef.type + '#' + str(to_evdef.value),
                    "child_name": to_evdef.value,
                }
            )
        return causality_records

class AlgorithmChoiceError(Exception):
    """Exception raised for errors when algorithm doesn't exist."""

    def __init__(self, message):
        self.message = message
    