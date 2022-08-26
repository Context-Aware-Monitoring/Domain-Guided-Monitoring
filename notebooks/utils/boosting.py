import pandas as pd
import numpy as np
import json
from src.refinement import KnowledgeProcessor, RefinementConfig
from typing import Dict, List, Tuple
from concurrent import futures
import ast
import pickle
from utils.percentiles import load_input_percentiles
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import re
from src.features.knowledge import BaseKnowledge
from concurrent import futures
from functools import partial
from src.refinement import RefinementConfig, KnowledgeProcessor
import tensorflow as tf
import random
from src.features import preprocessing
from src.features.preprocessing import mimic
from src.features import sequences
from src.features.sequences import transformer as trfm
from src.features import knowledge

def _calculate_edge_comparison_for_edge_unrestricted(
        edge: Tuple[str, str],
        processor: KnowledgeProcessor,
        attention_comp: Dict[str, Dict[str, float]],
        train_frequency: Dict[str, Dict[str, float]],
        comparison_df: pd.DataFrame,
        io_compatibility: Dict[Tuple[str, str], Tuple[float, float]],
        knowledge: BaseKnowledge):
        (c, p) = edge

        relevant_df = comparison_df[
            comparison_df["input_converted"].apply(lambda x: c in x)
        ]

        if processor.config.restrict_outputs_to_ancestors and knowledge is not None:
            relevant_df = relevant_df[
                relevant_df["output_converted"].apply(lambda x: knowledge.is_connected(c, x))
            ]
        
        if len(relevant_df) == 0:
            return None

        edge_weight = attention_comp.get(c, {}).get(p, -1)
        if float(edge_weight) < processor.config.min_edge_weight:
            return None

        frequency = train_frequency.get(c, {}).get("absolue_frequency", 0.0)
        if frequency > processor.config.max_train_examples:
            return None

        return {
            "child": c,
            "parent": p,
            "child_metric": processor._calculate_refinement_metric(
                c, relevant_df, io_compatibility
            )
        }

def _calculate_edge_comparison_unrestricted(
    processor: KnowledgeProcessor,
    attention_base: Dict[str, Dict[str, float]],
    attention_comp: Dict[str, Dict[str, float]],
    train_frequency: Dict[str, Dict[str, float]],
    comparison_df: pd.DataFrame,
    io_compatibility: Dict[Tuple[str, str], Tuple[float, float]],
    knowledge: BaseKnowledge = None
    ) -> pd.DataFrame:

    edges_base = set([(c, p) for c, ps in attention_base.items() for p in ps])
    edges_comp = set([(c, p) for c, ps in attention_comp.items() for p in ps])
    edges = edges_base.union(edges_comp)

    records = []
    with futures.ProcessPoolExecutor() as pool:
        for record in pool.map(
            partial(
                _calculate_edge_comparison_for_edge_unrestricted,
                processor=processor,
                attention_comp=attention_comp,
                train_frequency=train_frequency,
                comparison_df=comparison_df,
                io_compatibility=io_compatibility,
                knowledge=knowledge
            ), edges, chunksize=256
        ):
            if record is not None:
                records.append(record)

    metric_df = pd.DataFrame.from_records(
        records, columns=["child", "parent", "child_metric"]
    )

    if len(metric_df) > 0:
        parent_metric_df = metric_df.groupby("parent")["child_metric"].mean().reset_index(name="parent_metric")
        metric_df = metric_df.merge(parent_metric_df, on="parent")

    metric_df["refinement_metric"] = metric_df.apply(
        lambda x: processor._interpolate_metric(x["child_metric"], x["parent_metric"]),
        result_type="reduce", axis=1
    )

    return metric_df

def load_from_processor(index: int, processor: KnowledgeProcessor, reference_run_id: str, refinement_run_id: str):
    attention_base = processor._load_attention_weights(reference_run_id)
    attention_comp = processor._load_attention_weights(refinement_run_id)
    train_frequency = processor._load_input_frequency_dict(refinement_run_id)
    (comparison_df, io_compatibility) = processor._load_comparison_df(
        run_id_base=reference_run_id, run_id_comp=refinement_run_id
    )

    edge_comparison_df = _calculate_edge_comparison_unrestricted(
        processor=processor,
        attention_base=attention_base,
        attention_comp=attention_comp,
        train_frequency=train_frequency,
        comparison_df=comparison_df,
        io_compatibility=io_compatibility,
        knowledge=None # Ignore because we are not using it
    )

    if isinstance(processor.config.corrective_factor, float):
        # Handle legacy runs which don't use a list of corrective factors yet
        corrective_factor = processor.config.corrective_factor
    else:
        corrective_factor = processor._get_current_corrective_factor(processor.config.corrective_factor, index)

    edge_comparison_df["refinement_score"] = edge_comparison_df.apply(
        lambda x: processor._calculate_refinement_score(
            corrective_factor,
            x["refinement_metric"],
            attention_comp,
            x["child"], x["parent"]
        ),
        axis=1
    )

    return edge_comparison_df

def get_attention_for_run(run: str):
    with open("../gsim01/mlruns/1/{run}/artifacts/attention.json".format(run=run), "r") as attention_file:
        return json.load(attention_file)["attention_weights"]

def calculate_comparison(runs_df: pd.DataFrame, args: Tuple[int, str, str, str]):
    index, reference_id, before_id, after_id = args
    run = runs_df[runs_df["info_run_id"] == after_id]

    refinement_config = RefinementConfig()
    aggregate_metric_for_parents = run["data_params_RefinementConfigaggregate_metric_for_parents"].item() == 'True'
    contribution = float(run["data_params_RefinementConfigaggregated_parents_contribution"].fillna("0.0").item())
    if aggregate_metric_for_parents: # Handle legacy runs which don't use parent contribution yet
        contribution = 0.5
    refinement_config.aggregated_parents_contribution = contribution
    refinement_config.restrict_outputs_to_ancestors = run["data_params_RefinementConfigrestrict_outputs_to_ancestors"].item() == 'True'
    refinement_config.max_refinement_metric = float(run["data_params_RefinementConfigmax_refinement_metric"].item())
    refinement_config.max_train_examples = int(run["data_params_RefinementConfigmax_train_examples"].item())
    refinement_config.min_edge_weight = float(run["data_params_RefinementConfigmin_edge_weight"].item())
    refinement_config.refinement_metric_maxrank = int(run["data_params_RefinementConfigrefinement_metric_maxrank"].item())
    refinement_config.refinement_metric = run["data_params_RefinementConfigrefinement_metric"].item()
    refinement_config.mlflow_dir = "../gsim01/mlruns/1/"
    refinement_config.corrective_factor = ast.literal_eval(run["data_params_RefinementConfigcorrective_factor"].item())
    refinement_config.correction_attention_scale = float(run["data_params_RefinementConfigcorrection_attention_scale"].item())
    refinement_config.rank_decay_rate = float(run["data_params_RefinementConfigrank_decay_rate"].item())
    refinement_config.compatibility_factor = float(run["data_params_RefinementConfigcompatibility_factor"].item())

    processor = KnowledgeProcessor(refinement_config)

    with futures.ProcessPoolExecutor() as pool:
        after = pool.submit(load_from_processor, max(index, 0), processor, reference_id, after_id)

        if before_id == reference_id:
            edges_before_correction = pd.DataFrame(columns=["child", "parent", "refinement_metric"])
        else:
            before = pool.submit(load_from_processor, max(index, 0), processor, reference_id, before_id)
            edges_before_correction = before.result()
        edges_after_correction = after.result()

    edges_before_correction["corrected"] = edges_before_correction.apply(lambda x: x["refinement_metric"] < refinement_config.max_refinement_metric, result_type="reduce", axis=1)

    attention_before_correction = get_attention_for_run(before_id)
    attention_after_correction = get_attention_for_run(after_id)

    edges_before_correction["attention"] = edges_before_correction.apply(lambda x: attention_before_correction[x["child"]][x["parent"]], result_type="reduce", axis=1)
    edges_after_correction["attention"] = edges_after_correction.apply(lambda x: attention_after_correction[x["child"]][x["parent"]], result_type="reduce", axis=1)

    frequency_lookup = processor._load_input_frequency_dict(after_id)

    # Merge data
    df = edges_before_correction.merge(edges_after_correction, on=["child", "parent"], suffixes=["_before", "_after"], how="outer")
    df["change"] = df.apply(lambda x: x["refinement_metric_after"] - x["refinement_metric_before"], axis=1)
    df["attention_change"] = df.apply(lambda x: float(x["attention_after"]) - float(x["attention_before"]), axis=1)
    df["train_frequency"] = df.apply(lambda x: frequency_lookup.get(x["child"], {}).get("absolue_frequency", 0.0), axis=1)

    input_percentiles = load_input_percentiles(frequency_lookup, 10)

    df["input_percentile"] = df["child"].apply(lambda x: next(i for i in range(len(input_percentiles)) if x in input_percentiles[i]))

    if index >= 0:
        df["iteration"] = index

    return df

class GroupComparison:
    def __init__(self, comparison_df: pd.DataFrame, total_comparison_df: pd.DataFrame):
        self.comparison_df = comparison_df
        self.total_comparison_df = total_comparison_df

    comparison_df: pd.DataFrame
    total_comparison_df: pd.DataFrame

def load_group_comparisons(runs_df: pd.DataFrame, refinement_run_timestamp: str, generation: bool = False):
    data_path = Path("boosting_data/{}.pkl".format(refinement_run_timestamp))

    if data_path.is_file():
        with open(data_path, "rb") as f:
            return pickle.load(f)

    relevant_runs = list(runs_df[runs_df["refinement_run"] == refinement_run_timestamp]["info_run_id"])

    original_run_id = None
    reference_run_id = None
    refinement_run_ids = []

    for run in relevant_runs:
        run_type = str(runs_df[(runs_df["refinement_run"] == refinement_run_timestamp) & (runs_df["info_run_id"] == run)]["refinement_type"].item())
        
        if run_type == "original":
            original_run_id = run
        elif run_type == "reference":
            reference_run_id = run
        else:
            refinement_run_ids.append((run_type, run))

    refinement_run_ids = [run_id for _,run_id in sorted(refinement_run_ids, key=lambda x: x[0])]

    comparisons_df = pd.DataFrame()
    args = []
    before_id = original_run_id

    if not generation:
        for after_id in refinement_run_ids:
            index = refinement_run_ids.index(after_id)
            args.append((index, reference_run_id, before_id, after_id))
            before_id = after_id
    else:
        for i in range(np.floor_divide(len(refinement_run_ids), 2)):
            noisy_run = refinement_run_ids[2 * i]
            refined_run = refinement_run_ids[2 * i + 1]
            
            args.append((2 * i, before_id, before_id, noisy_run))
            args.append((2 * i + 1, before_id, noisy_run, refined_run))
            before_id = refined_run

    # for after_id in refinement_run_ids:
    #     index = refinement_run_ids.index(after_id)
    #     args.append((index, before_id, after_id))
    #     before_id = after_id

    # with futures.ProcessPoolExecutor() as pool:
    #     for comp in pool.map(calculate_comparison, args):
    #         comparisons_df = comparisons_df.append(comp)

    for arg in args:
        comp = calculate_comparison(runs_df, arg)
        comparisons_df = comparisons_df.append(comp)

    comparisons_df = comparisons_df.reset_index(drop=True)
    if not generation:
        total_comparison_df = calculate_comparison(runs_df, (-1, reference_run_id, original_run_id, refinement_run_ids[-1]))
    else:
        total_comparison_df = calculate_comparison(runs_df, (-1, original_run_id, original_run_id, refinement_run_ids[-1]))

    data = GroupComparison(comparisons_df, total_comparison_df)

    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    return data

def _match_legacy_refinement_type(x):
    match = re.search(r'^(\d+.\d+)_(\w+)$', x["refinement_type"])
    if match:
        return pd.Series([match.group(1), match.group(2)])
    else:
        return pd.Series([x["refinement_run"], x["refinement_type"]])

def resolve_references(mlflow_helper):
    relevant_run_df = mlflow_helper.run_df.copy()
    relevant_run_df["reference_id"] = relevant_run_df["data_tags_refinement_reference_id"]
    relevant_run_df["original_id"] = relevant_run_df["data_tags_refinement_original_id"]
    relevant_run_df["refinement_run"] = relevant_run_df["data_tags_refinement_run"]
    relevant_run_df["refinement_type"] = relevant_run_df["data_tags_refinement_type"].fillna("")

    relevant_run_df[["refinement_run", "refinement_type"]] = relevant_run_df.apply(_match_legacy_refinement_type, axis=1)

    refinement_runs_df = relevant_run_df[relevant_run_df["refinement_run"].fillna("").astype(str).apply(len) > 0].copy()
    potential_references_df = relevant_run_df[relevant_run_df["refinement_run"].fillna("") == ""]

    references_df = refinement_runs_df[["refinement_run", "reference_id", "original_id"]].melt(id_vars=["refinement_run"], value_vars=["reference_id", "original_id"], var_name="refinement_type", value_name="info_run_id").drop_duplicates()
    references_df["refinement_type"] = references_df["refinement_type"].replace(to_replace="reference_id", value="reference").replace(to_replace="original_id", value="original")

    result_df = pd.merge(potential_references_df.drop(columns=["refinement_run", "refinement_type"]), references_df, on="info_run_id")
    return result_df.append(refinement_runs_df).sort_values(by="refinement_run", ascending=True)

def plot_metric_change(data: pd.DataFrame):
    g = sns.relplot(data=data,
    x="refinement_metric_before",
    y="refinement_metric_after",
    hue="attention_change", palette=sns.color_palette("Reds", as_cmap=True),
    row="iteration")

    x_min = np.floor(data["refinement_metric_before"].min())
    x_max = np.ceil(data["refinement_metric_before"].max())

    for ax in g.axes.flat:
        ax.plot(np.linspace(x_min, x_max), np.linspace(x_min, x_max), 'k--')

    plt.show()

def plot_percentile_metrics(group: GroupComparison):
    df = group.total_comparison_df.copy(deep=True)
    all_df = group.comparison_df
    df["unseen"] = df["train_frequency"].apply(lambda x: x == 0)
    corrected_df = all_df.groupby(["child", "parent"])["corrected"].any().reset_index(name="corrected_in_any")
    df = df.merge(corrected_df, on=["child", "parent"])

    merged_metric_df = df.melt(id_vars=["unseen", "corrected_in_any", "input_percentile"], value_vars=["refinement_metric_before", "refinement_metric_after"])

    g = sns.relplot(data=merged_metric_df[merged_metric_df["corrected_in_any"]],
        col="unseen",
        col_order=[True, False],
        x="input_percentile",
        y="value",
        hue="variable",
        kind="line",
        height=8
    )
    
    g.set(xlabel="input_percentile", ylabel="refinement_metric")
    g.fig.suptitle("Before/after refinement metric per input-percentile among corrected and seen/unseen features", y=1.05, fontsize=12)
    plt.show()
    
    g = sns.lineplot(data=df,
        x="input_percentile",
        y="change",
        hue="corrected_in_any"
    )
    plt.title("Change in refinement metric per input-percentile among corrected/uncorrected features", y=1.1, fontsize=12)

    g = sns.catplot(data=df,
        col="corrected_in_any",
        col_order=[True, False],
        row="unseen",
        row_order=[False, True],
        x="input_percentile",
        y="change",
        kind="strip",
        height=8
    )
    g.fig.suptitle("Change in refinement metric per input-percentile among seen/unseen and corrected/uncorrected features", y=1.05, fontsize=12)

    sns.set_style("ticks", {"axes.grid": True})
    g = sns.catplot(data=merged_metric_df,
        col="corrected_in_any",
        col_order=[True, False],
        row="unseen",
        row_order=[False, True],
        x="input_percentile",
        y="value",
        hue="variable",
        kind="violin",
        split=True,
        inner="quartile",
        cut=0,
        height=8
    )
    g.fig.suptitle("Before/after refinement metric per input-percentile among seen/unseen and corrected/uncorrected features", y=1.05, fontsize=12)
    sns.set_style("ticks", {"axes.grid": False})

def _compute_self_attention(row):
    before = 1.0 - row["attention_before"]
    after = 1.0 - row["attention_after"]

    return pd.Series([before, after], index=["attention_before", "attention_after"])

def plot_attention_distribution(group: GroupComparison):
    df = group.total_comparison_df.copy(deep=True)
    all_df = group.comparison_df

    df["attention_before"] = pd.to_numeric(df["attention_before"])
    df["attention_after"] = pd.to_numeric(df["attention_after"])

    # Calculate self-attention
    df = df.append(df.groupby("child").sum(["attention_before", "attention_after"]).apply(_compute_self_attention, axis=1).reset_index())
    
    # Mark edges which are only corrected in some of the iterations
    corrected_df = all_df.groupby(["child", "parent"])["corrected"].any().reset_index(name="corrected_in_any")
    df = df.merge(corrected_df, on=["child", "parent"])

    df["corrected_in_any"] = df["corrected_in_any"].fillna(False)

    df["attention_rank_before"] = df.groupby("child")["attention_before"].rank("dense",ascending=False)
    max_rank = df["attention_rank_before"].max()

    # If we have more than 10 ranks, the diagrams don't work - switch to deciles
    use_deciles = max_rank > 10
    df["attention_rank_decile_before"] = df["attention_rank_before"].apply(lambda x: round(x / max_rank, ndigits=1) * 10)

    df = df.melt(id_vars=["corrected_in_any", "attention_rank_before", "attention_rank_decile_before"], value_vars=["attention_before", "attention_after"])

    g = sns.catplot(data=df,
        col="corrected_in_any",
        col_order=[True, False],
        x="attention_rank_before" if not use_deciles else "attention_rank_decile_before",
        y="value",
        hue="variable",
        kind="violin",
        split=True,
        inner="quartile",
        cut=0,
        height=8
    )
    g.set(xlabel="rank_before" if not use_deciles else "rank_decile_before", ylabel="attention")
    g.fig.suptitle("Before/after attention distribution based on rank {}before correction among corrected/uncorrected edges".format("" if not use_deciles else "deciles "), y=1.05, fontsize=12)

    g = sns.catplot(data=df,
        col="corrected_in_any",
        col_order=[True, False],
        x="attention_rank_before" if not use_deciles else "attention_rank_decile_before",
        y="value",
        hue="variable",
        kind="strip",
        dodge=True,
        height=8
    )
    g.set(xlabel="rank_before" if not use_deciles else "rank_decile_before", ylabel="attention")
    g.fig.suptitle("Before/after attention distribution based on rank {}before correction among corrected/uncorrected edges".format("" if not use_deciles else "deciles "), y=1.05, fontsize=12)

def gather_stats_for_artificial(df: pd.DataFrame, threshold: float = 0.01):
    df = df.copy(deep=True)

    with open("../data/gram_original_file_knowledge.json") as f:
        loaded_knowledge = json.load(f)

    original_knowledge: Dict[str, List[str]] = {}
    for child, conns in loaded_knowledge.items():
        original_knowledge["level_0#" + child] = list(map(lambda x: "level_0#" + x, conns))

    df["artificial"] = df.apply(lambda x: (x["child"] not in original_knowledge) or (x["parent"] not in original_knowledge[x["child"]]), axis=1)

    stats = {}
    
    num_artificial_ignored_before = len(df[df["artificial"] & (pd.to_numeric(df["attention_before"]) < threshold)])
    num_artificial_ignored_after = len(df[df["artificial"] & (pd.to_numeric(df["attention_after"]) < threshold)])
    num_real_ignored_before = len(df[~df["artificial"] & (pd.to_numeric(df["attention_before"]) < threshold)])
    num_real_ignored_after = len(df[~df["artificial"] & (pd.to_numeric(df["attention_after"]) < threshold)])

    num_connections = len(df)
    num_artificial = len(df[df["artificial"]])
    num_real = len(df[~df["artificial"]])

    percentage_artificial = round(num_artificial / num_connections * 100.0, ndigits=2)
    percentage_real = round(num_real / num_connections * 100.0, ndigits=2)

    percentage_artificial_ignored_before = round(num_artificial_ignored_before / num_artificial * 100.0, ndigits=2)
    percentage_artificial_ignored_after = round(num_artificial_ignored_after / num_artificial * 100.0, ndigits=2)
    percentage_real_ignored_before = round(num_real_ignored_before / num_real * 100.0, ndigits=2)
    percentage_real_ignored_after = round(num_real_ignored_after / num_real * 100.0, ndigits=2)

    stats["num_connections"] = num_connections
    stats["num_artificial"] = num_artificial
    stats["num_real"] = num_real
    stats["percentage_artificial"] = percentage_artificial
    stats["percentage_real"] = percentage_real
    stats["num_artificial_ignored_before"] = num_artificial_ignored_before
    stats["percentage_artificial_ignored_before"] = percentage_artificial_ignored_before
    stats["num_artificial_ignored_after"] = num_artificial_ignored_after
    stats["percentage_artificial_ignored_after"] = percentage_artificial_ignored_after
    stats["num_real_ignored_before"] = num_real_ignored_before
    stats["percentage_real_ignored_before"] = percentage_real_ignored_before
    stats["num_real_ignored_after"] = num_real_ignored_after
    stats["percentage_real_ignored_after"] = percentage_real_ignored_after

    stats["threshold"] = threshold

    return pd.DataFrame(stats, index=[0])

def gather_stats_for_group_artificial(df: pd.DataFrame, threshold: float = 0.01):
    iterations = int(df["iteration"].max().item())

    stats_df = pd.DataFrame()

    for i in range(iterations + 1):
        stats = gather_stats_for_artificial(df[df["iteration"] == i], threshold)
        stats["iteration"] = i
        stats_df = stats_df.append(stats)

    return stats_df.reset_index(drop=True)

def plot_artificial_attention_distribution(group: GroupComparison):
    df = group.total_comparison_df.copy(deep=True)
    all_df = group.comparison_df

    df["attention_before"] = pd.to_numeric(df["attention_before"])
    df["attention_after"] = pd.to_numeric(df["attention_after"])

    with open("../data/gram_original_file_knowledge.json") as f:
        loaded_knowledge = json.load(f)

    original_knowledge: Dict[str, List[str]] = {}
    for child, conns in loaded_knowledge.items():
        original_knowledge["level_0#" + child] = list(map(lambda x: "level_0#" + x, conns))

    df["artificial"] = df.apply(lambda x: (x["child"] not in original_knowledge) or (x["parent"] not in original_knowledge[x["child"]]), axis=1)
    
    # Mark edges which are only corrected in some of the iterations
    corrected_df = all_df.groupby(["child", "parent"])["corrected"].any().reset_index(name="corrected_in_any")
    df = df.merge(corrected_df, on=["child", "parent"])

    df["corrected_in_any"] = df["corrected_in_any"].fillna(False)

    df["attention_rank_before"] = df.groupby("child")["attention_before"].rank("dense",ascending=False)
    max_rank = df["attention_rank_before"].max()

    # If we have more than 10 ranks, the diagrams don't work - switch to deciles
    use_deciles = max_rank > 10
    df["attention_rank_decile_before"] = df["attention_rank_before"].apply(lambda x: round(x / max_rank, ndigits=1) * 10)

    df = df.melt(id_vars=["input_percentile", "corrected_in_any", "artificial", "attention_rank_before", "attention_rank_decile_before"], value_vars=["attention_before", "attention_after"])

    g = sns.relplot(data=df[df["corrected_in_any"]],
        col="artificial",
        col_order=[True, False],
        x="attention_rank_before" if not use_deciles else "attention_rank_decile_before",
        y="value",
        hue="variable",
        kind="line",
        height=8
    )
    
    g.set(xlabel="rank_before", ylabel="refinement_metric")
    g.fig.suptitle("Before/after attention based on rank {}among corrected and artificial/real edges".format("" if not use_deciles else "deciles "), y=1.05, fontsize=12)
    plt.show()

    g = sns.catplot(data=df,
        col="corrected_in_any",
        col_order=[True, False],
        row="artificial",
        row_order=[True, False],
        x="attention_rank_before" if not use_deciles else "attention_rank_decile_before",
        y="value",
        hue="variable",
        kind="violin",
        split=True,
        inner="quartile",
        cut=0,
        height=8
    )
    g.set(xlabel="rank_before" if not use_deciles else "rank_decile_before", ylabel="attention")
    g.fig.suptitle("Before/after attention distribution based on rank {}before correction among corrected/uncorrected edges".format("" if not use_deciles else "deciles "), y=1.05, fontsize=12)

    g = sns.catplot(data=df,
        col="corrected_in_any",
        col_order=[True, False],
        row="artificial",
        row_order=[True, False],
        x="attention_rank_before" if not use_deciles else "attention_rank_decile_before",
        y="value",
        hue="variable",
        kind="strip",
        dodge=True,
        height=8
    )
    g.set(xlabel="rank_before" if not use_deciles else "rank_decile_before", ylabel="attention")
    g.fig.suptitle("Before/after attention distribution based on rank {}before correction among corrected/uncorrected edges".format("" if not use_deciles else "deciles "), y=1.05, fontsize=12)

def plot_artificial_percentile_metrics(group: GroupComparison):
    df = group.total_comparison_df.copy(deep=True)
    all_df = group.comparison_df

    df["attention_before"] = pd.to_numeric(df["attention_before"])
    df["attention_after"] = pd.to_numeric(df["attention_after"])

    with open("../data/gram_original_file_knowledge.json") as f:
        loaded_knowledge = json.load(f)

    original_knowledge: Dict[str, List[str]] = {}
    for child, conns in loaded_knowledge.items():
        original_knowledge["level_0#" + child] = list(map(lambda x: "level_0#" + x, conns))

    df["artificial"] = df.apply(lambda x: (x["child"] not in original_knowledge) or (x["parent"] not in original_knowledge[x["child"]]), axis=1)

    corrected_df = all_df.groupby(["child", "parent"])["corrected"].any().reset_index(name="corrected_in_any")
    df = df.merge(corrected_df, on=["child", "parent"])

    merged_metric_df = df.melt(id_vars=["artificial", "corrected_in_any", "input_percentile"], value_vars=["refinement_metric_before", "refinement_metric_after"])

    g = sns.relplot(data=merged_metric_df[merged_metric_df["corrected_in_any"]],
        col="artificial",
        col_order=[True, False],
        x="input_percentile",
        y="value",
        hue="variable",
        kind="line",
        height=8
    )
    
    g.set(xlabel="input_percentile", ylabel="refinement_metric")
    g.fig.suptitle("Before/after refinement metric per input-percentile among corrected and artificial/real edges", y=1.05, fontsize=12)
    plt.show()
    
    g = sns.lineplot(data=df,
        x="input_percentile",
        y="change",
        hue="corrected_in_any"
    )
    plt.title("Change in refinement metric per input-percentile among corrected/uncorrected edges", y=1.1, fontsize=12)

    g = sns.catplot(data=df,
        col="corrected_in_any",
        col_order=[True, False],
        row="artificial",
        row_order=[False, True],
        x="input_percentile",
        y="change",
        kind="strip",
        height=8
    )
    g.fig.suptitle("Change in refinement metric per input-percentile among artificial/real and corrected/uncorrected edges", y=1.05, fontsize=12)

    sns.set_style("ticks", {"axes.grid": True})
    g = sns.catplot(data=merged_metric_df,
        col="corrected_in_any",
        col_order=[True, False],
        row="artificial",
        row_order=[False, True],
        x="input_percentile",
        y="value",
        hue="variable",
        kind="violin",
        split=True,
        inner="quartile",
        cut=0,
        height=8
    )
    g.fig.suptitle("Before/after refinement metric per input-percentile among artificial/real and corrected/uncorrected edges", y=1.05, fontsize=12)
    sns.set_style("ticks", {"axes.grid": False})

def gather_stats(df: pd.DataFrame, threshold: float = 0.0):
    stats = {}

    num_improved = len(df[df["change"] > threshold])
    num_regressed = len(df[df["change"] < -threshold])
    num_profitable_before = len(df[df["refinement_metric_before"] >= threshold])
    num_profitable_after = len(df[df["refinement_metric_after"] >= threshold])
    num_connections = len(df)
    percentage_improved = round(num_improved / num_connections * 100.0, ndigits=2)
    percentage_regressed = round(num_regressed / num_connections * 100.0, ndigits=2)
    percentage_profitable_before = round(num_profitable_before / num_connections * 100.0, ndigits=2)
    percentage_profitable_after = round(num_profitable_after / num_connections * 100.0, ndigits=2)

    num_switched_harmful_helpful = len(df[(df["refinement_metric_before"] < -threshold) & (df["refinement_metric_after"] >= threshold)])
    num_switched_helpful_harmful = len(df[(df["refinement_metric_before"] >= threshold) & (df["refinement_metric_after"] < -threshold)])
    percentage_switched_harmful_helpful = round(num_switched_harmful_helpful / num_connections * 100.0, ndigits=2)
    percentage_switched_helpful_harmful = round(num_switched_helpful_harmful / num_connections * 100.0, ndigits=2)

    num_corrected_improved = len(df[df["corrected"] & (df["change"] > threshold)])
    num_corrected_regressed = len(df[df["corrected"] & (df["change"] < -threshold)])
    num_uncorrected_improved = len(df[~df["corrected"] & (df["change"] > threshold)])
    num_uncorrected_regressed = len(df[~df["corrected"] & (df["change"] < -threshold)])
    num_corrected = len(df[df["corrected"]])
    num_uncorrected = len(df[~df["corrected"]])
    percentage_corrected_improved = round(num_corrected_improved / num_corrected * 100.0, ndigits=2)
    percentage_corrected_improved_total = round(num_corrected_improved / num_connections * 100.0, ndigits=2)
    percentage_corrected_regressed = round(num_corrected_regressed / num_corrected * 100.0, ndigits=2)
    percentage_corrected_regressed_total = round(num_corrected_regressed / num_connections * 100.0, ndigits=2)
    percentage_uncorrected_improved = round(num_uncorrected_improved / num_uncorrected * 100.0, ndigits=2)
    percentage_uncorrected_improved_total = round(num_uncorrected_improved / num_connections * 100.0, ndigits=2)
    percentage_uncorrected_regressed = round(num_uncorrected_regressed / num_uncorrected * 100.0, ndigits=2)
    percentage_uncorrected_regressed_total = round(num_uncorrected_regressed / num_connections * 100.0, ndigits=2)

    num_ignored_before = len(df[(pd.to_numeric(df["attention_before"]) < 0.01)])
    num_ignored_after = len(df[(pd.to_numeric(df["attention_after"]) < 0.01)])

    percentage_ignored_before = round(num_ignored_before / num_connections * 100.0, ndigits=2)
    percentage_ignored_after = round(num_ignored_after / num_connections * 100.0, ndigits=2)

    avg_overall_change = round(df[df["change"].abs() > threshold]["change"].mean(), ndigits=4)
    avg_uncorrected_change = round(df[~df["corrected"] & (df["change"].abs() > threshold)]["change"].mean(), ndigits=4)
    avg_corrected_change = round(df[df["corrected"] & (df["change"].abs() > threshold)]["change"].mean(), ndigits=4)

    stats["num_connections"] = num_connections
    stats["num_corrected"] = num_corrected
    stats["num_uncorrected"] = num_uncorrected
    
    stats["num_improved"] = num_improved
    stats["percentage_improved"] = percentage_improved
    stats["num_worsened"] = num_regressed
    stats["percentage_worsened"] = percentage_regressed

    stats["num_profitable_before"] = num_profitable_before
    stats["percentage_profitable_before"] = percentage_profitable_before
    stats["num_profitable_after"] = num_profitable_after
    stats["percentage_profitable_after"] = percentage_profitable_after

    stats["num_switched_harmful_helpful"] = num_switched_harmful_helpful
    stats["percentage_switched_harmful_helpful"] = percentage_switched_harmful_helpful
    stats["num_switched_helpful_harmful"] = num_switched_helpful_harmful
    stats["percentage_switched_helpful_harmful"] = percentage_switched_helpful_harmful

    stats["num_corrected_improved"] = num_corrected_improved
    stats["percentage_corrected_improved"] = percentage_corrected_improved
    stats["percentage_corrected_improved_total"] = percentage_corrected_improved_total
    stats["num_corrected_worsened"] = num_corrected_regressed
    stats["percentage_corrected_worsened"] = percentage_corrected_regressed
    stats["percentage_corrected_worsened_total"] = percentage_corrected_regressed_total

    stats["num_uncorrected_improved"] = num_uncorrected_improved
    stats["percentage_uncorrected_improved"] = percentage_uncorrected_improved
    stats["percentage_uncorrected_improved_total"] = percentage_uncorrected_improved_total
    stats["num_uncorrected_worsened"] = num_uncorrected_regressed
    stats["percentage_uncorrected_worsened"] = percentage_uncorrected_regressed
    stats["percentage_uncorrected_worsened_total"] = percentage_uncorrected_regressed_total

    stats["num_ignored_before"] = num_ignored_before
    stats["percentage_ignored_before"] = percentage_ignored_before
    stats["num_ignored_after"] = num_ignored_after
    stats["percentage_ignored_after"] = percentage_ignored_after

    stats["avg_overall_change"] = avg_overall_change
    stats["avg_uncorrected_change"] = avg_uncorrected_change
    stats["avg_corrected_change"] = avg_corrected_change

    stats["threshold"] = threshold

    return pd.DataFrame(stats, index=[0])

def gather_stats_for_group(df: pd.DataFrame, threshold: float = 0.0):
    iterations = int(df["iteration"].max().item())

    stats_df = pd.DataFrame()

    for i in range(iterations + 1):
        stats = gather_stats(df[df["iteration"] == i], threshold)
        stats["iteration"] = i
        stats_df = stats_df.append(stats)

    return stats_df.reset_index(drop=True)

def aggregate_stats_for_groups(groups, threshold: float = 0.0):
    stats_df = pd.DataFrame()

    for (index, group) in groups:
        stats = gather_stats(group.total_comparison_df, threshold)
        stats["group"] = index
        stats_df = stats_df.append(stats)

    return stats_df

def load_hierarchy():
    tensorflow_seed = 7796
    random_seed = 82379498237

    tf.random.set_seed(tensorflow_seed)
    random.seed(random_seed)

    preprocessor_config = mimic.MimicPreprocessorConfig()
    preprocessor_config.prediction_column = "level_0"
    preprocessor_config.sequence_column_name = "level_all"

    sequences_df = preprocessing.MimicPreprocessor(preprocessor_config).load_data() 

    sequence_column_name = preprocessor_config.sequence_column_name

    transformer_config = sequences.SequenceConfig()
    transformer_config.x_sequence_column_name = "level_0"
    transformer_config.y_sequence_column_name = "level_3"
    transformer_config.predict_full_y_sequence_wide = True

    transformer = trfm.NextPartialSequenceTransformerFromDataframe(transformer_config)

    metadata = transformer.collect_metadata(sequences_df, sequence_column_name)   

    hierarchy_df = preprocessing.ICD9HierarchyPreprocessor(preprocessor_config).load_data()
    hierarchy_mapping_df = preprocessing.ICD9DataPreprocessor(preprocessor_config.icd9_file, preprocessor_config.icd9_hierarchy_file).load_data_as_hierarchy()
    hierarchy = knowledge.HierarchyKnowledge(knowledge.KnowledgeConfig())
    hierarchy.build_hierarchy_from_df(hierarchy_df, metadata.x_vocab, hierarchy_mapping_df, transformer_config.x_sequence_column_name, transformer_config.y_sequence_column_name)

    return hierarchy, hierarchy_mapping_df