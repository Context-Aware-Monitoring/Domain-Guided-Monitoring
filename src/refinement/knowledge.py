from .config import RefinementConfig
import json
from typing import Dict, List, Tuple, Set
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from src.runner import RunState
from src.features.knowledge import BaseKnowledge
from concurrent import futures
from functools import partial
import pickle

class KnowledgeProcessor:
    def __init__(self, config: RefinementConfig):
        self.config = config
        self.max_rank = 0

    def load_original_knowledge(self) -> Dict[str, List[str]]:
        with open(self.config.original_file_knowledge) as knowledge_file:
            return json.load(knowledge_file)

    def load_reference_knowledge(self) -> Dict[str, List[str]]:
        with open(self.config.reference_file_knowledge) as knowledge_file:
            return json.load(knowledge_file)

    def _calculate_added_edges(
        self,
        attention_base: Dict[str, Dict[str, float]],
        attention_comp: Dict[str, Dict[str, float]],
    ) -> Set[Tuple[str, str]]:
        edges_base = set([(c, p) for c, ps in attention_base.items() for p in ps])
        edges_comp = set([(c, p) for c, ps in attention_comp.items() for p in ps])
        logging.debug(
            "Found %d edges in base, %d in comp", len(edges_base), len(edges_comp)
        )
        return edges_comp - edges_base

    def _make_outlier_score(self, rank_before: int, rank_after: int):
        decay = np.exp(-self.config.rank_decay_rate * min(rank_before, rank_after) / self.max_rank)
        return (rank_before - rank_after) / np.sqrt(2) * decay

    def _calculate_refinement_metric(
        self, input_feature: str, comparison_df: pd.DataFrame
    ) -> float:
        # refinement_metric > 0 -> comparison is better than base
        relevant_df = comparison_df.copy()
        if len(relevant_df) == 0:
            return 0
        if self.config.refinement_metric_maxrank > 0:
            relevant_df["output_rank_base"] = relevant_df["output_rank_base"].apply(
                lambda x: min(x, self.config.refinement_metric_maxrank)
            )
            relevant_df["output_rank_comp"] = relevant_df["output_rank_comp"].apply(
                lambda x: min(x, self.config.refinement_metric_maxrank)
            )

        if "outlier_score" in self.config.refinement_metric:
            outlier_scores = (
                relevant_df[["output_rank_base", "output_rank_comp"]]
                .apply(lambda x: self._make_outlier_score(int(x[0]), int(x[1])), axis=1)
                .to_list()
            )
            if "median" in self.config.refinement_metric:
                return np.median(outlier_scores)
            elif "mean" in self.config.refinement_metric:
                return np.mean(outlier_scores)
        elif "accuracy" in self.config.refinement_metric:
            accuracy_ats = [
                int(s) for s in self.config.refinement_metric.split("_") if s.isdigit()
            ]
            accuracy_at = accuracy_ats[0] if len(accuracy_ats) > 0 else 1
            accuracy_base = len(
                relevant_df[relevant_df["output_rank_base"] < accuracy_at]
            ) / len(relevant_df)
            accuracy_comp = len(
                relevant_df[relevant_df["output_rank_comp"] < accuracy_at]
            ) / len(relevant_df)
            return accuracy_comp - accuracy_base

        logging.error("Unknown refinement metric: %s", self.config.refinement_metric)
        return 0

    def _calculate_edge_comparison_for_edge(
        self,
        edge: Tuple[str, str],
        attention_comp: Dict[str, Dict[str, float]],
        train_frequency: Dict[str, Dict[str, float]],
        comparison_df: pd.DataFrame,
        knowledge: BaseKnowledge):
        (c, p) = edge

        if c == p:
            return None

        relevant_df = comparison_df[
            comparison_df["inputs"].apply(lambda x: c + "," in x)
        ]

        if self.config.restrict_outputs_to_ancestors:
            relevant_df = relevant_df[
                relevant_df["output"].apply(lambda x: knowledge.is_connected(c, x))
            ]
        
        if len(relevant_df) == 0:
            return None

        edge_weight = attention_comp.get(c, {}).get(p, -1)
        if float(edge_weight) < self.config.min_edge_weight:
            return None

        frequency = train_frequency.get(c, {}).get("absolue_frequency", 0.0)
        if frequency > self.config.max_train_examples:
            return None

        return {
            "child": c,
            "parent": p,
            "child_metric": self._calculate_refinement_metric(
                c, relevant_df
            )
        }

    def _interpolate_metric(self, child_metric: float, parent_metric: float) -> float:
        contribution = self.config.aggregated_parents_contribution
        return (1.0 - contribution) * child_metric + contribution * parent_metric

    def _calculate_edge_comparison(
        self,
        attention_base: Dict[str, Dict[str, float]],
        attention_comp: Dict[str, Dict[str, float]],
        train_frequency: Dict[str, Dict[str, float]],
        comparison_df: pd.DataFrame,
        knowledge: BaseKnowledge
    ) -> pd.DataFrame:
        added_edges = self._calculate_added_edges(
            attention_base=attention_base, attention_comp=attention_comp
        )

        records = []
        with futures.ProcessPoolExecutor() as pool:
            for record in pool.map(
                partial(
                    self._calculate_edge_comparison_for_edge,
                    attention_comp=attention_comp,
                    train_frequency=train_frequency,
                    comparison_df=comparison_df,
                    knowledge=knowledge
                ), added_edges, chunksize=256
            ):
                if record is not None:
                    records.append(record)

        metric_df = pd.DataFrame.from_records(
            records, columns=["child", "parent", "child_metric"]
        )

        parent_metric_df = metric_df.groupby("parent")["child_metric"].mean().reset_index(name="parent_metric")
        metric_df = metric_df.merge(parent_metric_df, on="parent")
        metric_df["refinement_metric"] = metric_df.apply(
            lambda x: self._interpolate_metric(x["child_metric"], x["parent_metric"]),
            axis=1
        )

        return metric_df


    def load_refined_knowledge(
        self, refinement_run_id: str, reference_run_id: str
    ) -> Dict[str, List[str]]:
        attention_base = self._load_attention_weights(reference_run_id)
        attention_comp = self._load_attention_weights(refinement_run_id)
        train_frequency = self._load_input_frequency_dict(refinement_run_id)
        comparison_df = self._load_comparison_df(
            run_id_base=reference_run_id, run_id_comp=refinement_run_id
        )

        edge_comparison_df = self._calculate_edge_comparison(
            attention_base=attention_base,
            attention_comp=attention_comp,
            train_frequency=train_frequency,
            comparison_df=comparison_df,
        )
        edge_comparison_df = (
            edge_comparison_df[
                edge_comparison_df["refinement_metric"]
                < self.config.max_refinement_metric
            ]
            .sort_values(by="refinement_metric", ascending=True)
            .head(n=self.config.max_edges_to_remove)
        )
        refined_knowledge = {c: [c] for c in attention_comp}
        for child, parents in attention_comp.items():
            for parent in parents:
                if (
                    len(
                        edge_comparison_df[
                            (edge_comparison_df["child"] == child)
                            & (edge_comparison_df["parent"] == parent)
                        ]
                    )
                    > 0
                ):
                    continue

                refined_knowledge[child].append(parent)

        return refined_knowledge

    def _calculate_refinement_score(
        self,
        corrective_factor: float,
        refinement_metric: float,
        attention: Dict[str, Dict[str, float]],
        child: str, parent: str
    ):
        adjustment = np.exp(corrective_factor * refinement_metric)
        weight = self.config.correction_attention_scale * float(attention[child][parent])

        return adjustment * (1.0 - weight) / (1.0 - adjustment * weight)

    def _calculate_edge_comparison_for_runs(
        self,
        reference_run_id,
        refinement_run_id,
        knowledge: BaseKnowledge
    ):
        attention_base = self._load_attention_weights(reference_run_id)
        attention_comp = self._load_attention_weights(refinement_run_id)
        train_frequency = self._load_input_frequency_dict(refinement_run_id)
        comparison_df = self._load_comparison_df(
            run_id_base=reference_run_id, run_id_comp=refinement_run_id
        )

        return self._calculate_edge_comparison(
            attention_base=attention_base,
            attention_comp=attention_comp,
            train_frequency=train_frequency,
            comparison_df=comparison_df,
            knowledge=knowledge
        )

    def update_corrective_terms(
        self,
        index: int,
        current_run_id: str,
        run: RunState,
        refinement_run_id: str,
        reference_run_id: str
    ):
        logging.info("Starting update of corrective terms")
        edge_comparison_df = self._calculate_edge_comparison_for_runs(
            reference_run_id=reference_run_id,
            refinement_run_id=refinement_run_id,
            knowledge=run.knowledge
        )

        edge_comparison_df = (
            edge_comparison_df[
                edge_comparison_df["refinement_metric"] # Only allow edges that decrease performance
                < self.config.max_refinement_metric
            ]
            .sort_values(by="refinement_metric", ascending=True)
            .head(n=self.config.max_edges_to_remove)
        )
        
        attention_comp = self._load_attention_weights(refinement_run_id)
        edge_comparison_df["refinement_score"] = edge_comparison_df.apply(
            lambda x: self._calculate_refinement_score(
                self._get_current_corrective_factor(self.config.corrective_factor, index),
                x["refinement_metric"],
                attention_comp,
                x["child"], x["parent"]
            ),
            axis=1
        )

        edges_to_correct: Dict[Tuple[str, str], float] = {}

        for _, row in edge_comparison_df.iterrows():
            edges_to_correct[(row["child"], row["parent"])] = row["refinement_score"]

        logging.info("Updating corrective terms for %d edges", len(edges_to_correct))

        file_path = self.config.mlflow_dir + "{run_id}/artifacts/".format(run_id=current_run_id)
        
        edge_comparison_df.to_pickle(file_path + "edge_comparison.pkl")
        corrective_terms = run.model.embedding_layer.get_corrective_terms_as_list()
        with open(file_path + "previous_corrective_terms.pkl", "wb") as f:
            pickle.dump(corrective_terms, f)

        run.model.embedding_layer.update_corrective_terms(edges_to_correct)

        run.model.rnn_layer.trainable = self._is_trainable(self.config.freeze_rnn_sequence, index)
        run.model.embedding_layer.trainable = self._is_trainable(self.config.freeze_embeddings_sequence, index)
        run.model.activation_layer.trainable = self._is_trainable(self.config.freeze_activation_sequence, index)

    def restore_best_corrective_terms(
        self,
        run: RunState,
        refinement_run_id: str,
        reference_run_id: str
    ):
        logging.info("Restoring best corrective terms")
        file_path = self.config.mlflow_dir + "{run_id}/artifacts/".format(run_id=refinement_run_id)

        with open(file_path + "previous_corrective_terms.pkl", "rb") as f:
            corrective_terms = pickle.load(f)
        run.model.embedding_layer.set_corrective_terms_from_list(corrective_terms)

        edge_comparison_before_df = pd.read_pickle(file_path + "edge_comparison.pkl") 
        edge_comparison_after_df = self._calculate_edge_comparison_for_runs(
            reference_run_id=reference_run_id,
            refinement_run_id=refinement_run_id,
            knowledge=run.knowledge
        )

        edge_comparison_df = edge_comparison_before_df.merge(
            edge_comparison_after_df,
            on=["child", "parent"], suffixes=["_before", "_after"]
        )
        edge_comparison_df = (
            edge_comparison_df[
                edge_comparison_df["refinement_metric_before"] # Only allow edges that decrease performance
                < self.config.max_refinement_metric
            ]
            .sort_values(by="refinement_metric_before", ascending=True)
            .head(n=self.config.max_edges_to_remove)
        )

        previously_corrected_count = len(edge_comparison_df)

        # Re-apply all corrections which were helpful on top of previous corrections
        edge_comparison_df = edge_comparison_df[edge_comparison_df.apply(
            lambda x: (
                x["refinement_metric_before"] - x["refinement_metric_after"]
            ) <= self.config.restore_threshold, axis=1
        )]

        logging.info("Reapplying corrections for %d edges (restoring %d edges)",
         len(edge_comparison_df), previously_corrected_count - len(edge_comparison_df)
        )

        edges_to_reapply: Dict[Tuple[str, str], float] = {}

        for _, row in edge_comparison_df.iterrows():
            edges_to_reapply[(row["child"], row["parent"])] = row["refinement_score"]

        run.model.embedding_layer.update_corrective_terms(edges_to_reapply)

    def _is_trainable(self, sequence: List[int], index: int) -> bool:
        # Sequence stores True (=1) if frozen, therefore negate
        return False if sequence[min(index, len(sequence) - 1)] == 1 else True

    def _get_current_corrective_factor(self, sequence: List[float], index: int) -> float:
        return sequence[min(index, len(sequence) - 1)]

    def _load_attention_weights(self, run_id):
        attention_path = Path(
            self.config.mlflow_dir
            + "{run_id}/artifacts/attention.json".format(run_id=run_id)
        )
        if not attention_path.exists():
            logging.debug(
                "No attention file for run {} in local MlFlow dir".format(run_id)
            )
            return {}

        with open(attention_path) as attention_file:
            return json.load(attention_file)["attention_weights"]

    def _get_best_rank_of(self, output: str, predictions_str: str) -> int:
        predictions = ast.literal_eval(predictions_str)
        self.max_rank = max(self.max_rank, len(predictions))
        return len([x for x in predictions if predictions[x] > predictions[output]])

    def _convert_prediction_df(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        prediction_df["input_converted"] = prediction_df["input"].apply(
            lambda x: " -> ".join(
                [
                    ", ".join([str(val) for val in sorted(v)])
                    for (_, v) in sorted(
                        ast.literal_eval(x).items(), key=lambda y: y[0]
                    )
                ]
            )
        )
        prediction_df["inputs"] = prediction_df["input"].apply(
            lambda x: ",".join(
                sorted(
                    set(
                        [
                            x
                            for xs in [
                                [str(val) for val in sorted(v)]
                                for (_, v) in sorted(
                                    ast.literal_eval(x).items(), key=lambda y: y[0]
                                )
                            ]
                            for x in xs
                        ]
                    )
                )
            )
            + ","
        )
        prediction_df["output"] = prediction_df["output"].apply(
            lambda x: ast.literal_eval(x)
        )
        prediction_df = prediction_df.explode("output")
        prediction_df["output_rank"] = prediction_df[["output", "predictions"]].apply(
            lambda x: self._get_best_rank_of(x[0], x[1]), axis=1
        )
        return prediction_df

    def _load_prediction_df(self, run_id) -> pd.DataFrame:
        run_prediction_output_path = Path(
            self.config.mlflow_dir
            + "{run_id}/artifacts/prediction_output.csv".format(run_id=run_id)
        )
        if not run_prediction_output_path.exists():
            logging.debug(
                "No prediction output file for run {} in local MlFlow dir".format(
                    run_id
                )
            )
            return pd.DataFrame()

        prediction_output_df = pd.read_csv(run_prediction_output_path)
        return self._convert_prediction_df(prediction_output_df)

    def _load_input_frequency_dict(self, run_id) -> Dict[str, Dict[str, float]]:
        run_frequency_path = Path(
            self.config.mlflow_dir
            + "{run_id}/artifacts/train_frequency.csv".format(run_id=run_id)
        )
        if not run_frequency_path.exists():
            logging.debug("No frequency file for run {} in MlFlow dir".format(run_id))
            return {}

        input_frequency_df = pd.read_csv(run_frequency_path).set_index("feature")
        input_frequency_df["relative_frequency"] = input_frequency_df[
            "absolue_frequency"
        ] / sum(input_frequency_df["absolue_frequency"])
        return input_frequency_df.to_dict("index")

    def _load_comparison_df(
        self, run_id_base, run_id_comp, suffix_base="_base", suffix_comp="_comp"
    ) -> pd.DataFrame:
        prediction_df_base = self._load_prediction_df(run_id_base)
        prediction_df_comp = self._load_prediction_df(run_id_comp)
        if prediction_df_base is None or prediction_df_comp is None:
            logging.error(
                "Unable to load prediction_dfs for runs {} and {}".format(
                    run_id_base, run_id_comp
                )
            )
            return pd.DataFrame()

        comparison_df = pd.merge(
            prediction_df_base.sort_values(by=["input_converted", "inputs", "output"])
            .reset_index(drop=True)
            .reset_index(drop=False),
            prediction_df_comp.sort_values(by=["input_converted", "inputs", "output"])
            .reset_index(drop=True)
            .reset_index(drop=False),
            on=["index", "input_converted", "inputs", "output"],
            suffixes=(suffix_base, suffix_comp),
        )
        return comparison_df
