import logging
from src import _main
from src import features, refinement
import json
import time
from typing import Dict, List, Tuple
from mlflow.tracking import MlflowClient
import random
from src.main import _log_all_configs_to_mlflow
from src import ExperimentRunner, RunState, ExperimentConfig
import mlflow
import pickle
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib.font_manager").disabled = True


def _write_file_knowledge(knowledge: Dict[str, List[str]]):
    knowledge_config = features.knowledge.KnowledgeConfig()
    with open(knowledge_config.file_knowledge, "w") as knowledge_file:
        json.dump(knowledge, knowledge_file)


def calculate_num_connections(knowledge: Dict[str, List[str]]) -> int:
    return len(set([(x, con) for x, cons in knowledge.items() for con in set(cons)]))


def _write_reference_knowledge(refinement_config: refinement.RefinementConfig) -> int:
    logging.info("Writing reference knowledge...")
    reference_knowledge = refinement.KnowledgeProcessor(
        refinement_config
    ).load_reference_knowledge()    
    _write_file_knowledge(reference_knowledge)
    return calculate_num_connections(reference_knowledge)

def _add_random_connections(
    original_knowledge: Dict[str, List[str]],
    percentage: float = 0.1,
    generator = np.random.default_rng(ExperimentConfig().random_seed)
) -> Dict[str, List[str]]:
    connections = set([(child, parent) for child, parents in original_knowledge.items() for parent in parents if child != parent])
    children = list(set([x[0] for x in connections]))
    parents = list(set([x[1] for x in connections]))
    potential_connections = [
        (c, p) for c in children for p in parents if (c,p) not in connections and c != p
    ]
    connections_to_add_array = generator.choice(
        potential_connections,
        min(len(potential_connections), int(percentage * len(connections))),
        replace=False
    )
    connections_to_add = [tuple(x) for x in connections_to_add_array]

    logging.debug("Added %d connections from originally %d connections, %d children, %d parents", len(connections_to_add), len(connections), len(children), len(parents))
    updated_knowledge: Dict[str, List[str]] = {}
    for connection in set(connections_to_add).union(connections):
        updated_knowledge[connection[0]] = updated_knowledge.get(connection[0], []) + [connection[1]]
    return updated_knowledge


def _write_original_knowledge(refinement_config: refinement.RefinementConfig) -> int:
    logging.info("Writing original knowledge...")
    original_knowledge = refinement.KnowledgeProcessor(
        refinement_config
    ).load_original_knowledge()
    if refinement_config.edges_to_add[0] > 0:
        logging.info("Adding %f noise to original knowledge", refinement_config.edges_to_add[0])
        original_knowledge = _add_random_connections(original_knowledge, refinement_config.edges_to_add[0])
    _write_file_knowledge(original_knowledge)
    return calculate_num_connections(original_knowledge)


def _write_refined_knowledge(
    refinement_config: refinement.RefinementConfig,
    refinement_run_id: str,
    reference_run_id: str,
) -> int:
    logging.info("Writing refined knowledge...")
    refined_knowledge = refinement.KnowledgeProcessor(
        refinement_config
    ).load_refined_knowledge(
        refinement_run_id=refinement_run_id, reference_run_id=reference_run_id
    )
    _write_file_knowledge(refined_knowledge)
    return calculate_num_connections(refined_knowledge)


def _add_mlflow_tag(run_id: str, refinement_timestamp: int, suffix: str):
    mlflow_client = MlflowClient()
    mlflow_client.set_tag(
        run_id=run_id,
        key="refinement_type",
        value="{identifier}_{suffix}".format(
            identifier=str(refinement_timestamp), suffix=suffix
        ),
    )


def _add_mlflow_tags_for_new_run(run_id: str, refinement_timestamp: int, type: str):
    mlflow_client = MlflowClient()
    mlflow_client.set_tag(
        run_id=run_id,
        key="refinement_run",
        value=str(refinement_timestamp)
    )
    mlflow_client.set_tag(
        run_id=run_id,
        key="refinement_type",
        value=type
    )


def _add_mlflow_tags_for_refinement(run_id: str, refinement_timestamp: int, index: int, config: refinement.RefinementConfig):
    _add_mlflow_tags_for_new_run(run_id, refinement_timestamp, "refinement_{index}".format(index=str(index)))
    mlflow_client = MlflowClient()
    mlflow_client.set_tag(
        run_id=run_id,
        key="refinement_reference_id",
        value=config.reference_run_id
    )
    mlflow_client.set_tag(
        run_id=run_id,
        key="refinement_original_id",
        value=config.original_run_id
    )


def _do_reference_run(timestamp: int, config: refinement.RefinementConfig) -> str:
    if len(config.reference_run_id) > 0:
        return config.reference_run_id
    else:
        _write_reference_knowledge(config)
        reference_run_id = _main()
        _add_mlflow_tags_for_new_run(reference_run_id, timestamp, "reference")
        return reference_run_id


def _do_original_run(timestamp: int, config: refinement.RefinementConfig) -> Tuple[str, RunState]:
    if len(config.original_run_id) > 0:
        original_run_id = config.original_run_id
        original_knowledge = _get_knowledge_from_id(original_run_id)
        _write_file_knowledge(original_knowledge)

        state = ExperimentRunner().prepare_run()
        if config.keep_state_from_original:
            weights_path = (
                config.mlflow_dir
                + "{run_id}/artifacts/trained_weights.h5".format(run_id=original_run_id)
            )
            state.model.prediction_model.load_weights(weights_path)
        return (original_run_id, state)
    else:
        _write_original_knowledge(config)
        with mlflow.start_run() as run:
            original_run_id = run.info.run_id
            _log_all_configs_to_mlflow()
            runner = ExperimentRunner()
            state = runner.run(original_run_id)
            _add_mlflow_tags_for_new_run(original_run_id, timestamp, "original")

        if not config.keep_state_from_original:
            state = runner.prepare_run()
        return (original_run_id, state)


def main_boosting(refinement_config: refinement.RefinementConfig):
    mlflow.set_experiment("Domain Guided Monitoring")

    refinement_timestamp = time.time()

    reference_run_id = _do_reference_run(refinement_timestamp, refinement_config)
    (original_run_id, state) = _do_original_run(refinement_timestamp, refinement_config)

    refinement_run_ids = [original_run_id]

    processor = refinement.KnowledgeProcessor(refinement_config)
    runner = ExperimentRunner()

    for i in range(refinement_config.num_refinements):
        with mlflow.active_run() or mlflow.start_run() as run:
            current_run_id = run.info.run_id
            processor.update_corrective_terms(i, current_run_id, state, refinement_run_ids[-1], reference_run_id)
            _log_all_configs_to_mlflow()
            state = runner.run_from_state(current_run_id, state)
            index = 2 * i if refinement_config.restore_best_correction else i
            _add_mlflow_tags_for_refinement(current_run_id, refinement_timestamp, index, refinement_config)
            refinement_run_ids.append(current_run_id)

        if refinement_config.restore_best_correction:
            with mlflow.start_run() as run:
                current_run_id = run.info.run_id
                processor.restore_best_corrective_terms(state, refinement_run_ids[-1], reference_run_id)
                _log_all_configs_to_mlflow()
                state = runner.run_from_state(current_run_id, state)
                _add_mlflow_tags_for_refinement(current_run_id, refinement_timestamp, 2 * i + 1, refinement_config)
                refinement_run_ids.append(current_run_id)

        logging.info(
            "Completed boosting iteration {current} of {total}"
            .format(current=i+1, total=refinement_config.num_refinements)
        )

    logging.info("Finished boosting run ({group})".format(group=refinement_timestamp))
    logging.info("reference run id: {reference_run_id}".format(reference_run_id=reference_run_id))
    logging.info("original run id: {original_run_id}".format(original_run_id=original_run_id))

    for run in refinement_run_ids[1:]:
        logging.info("refinement run id: {refinement_run_id}".format(refinement_run_id=run))


def main_refinement(refinement_config: refinement.RefinementConfig):
    refinement_timestamp = time.time()
    num_reference_connections = _write_reference_knowledge(refinement_config)
    reference_run_id = _main()
    _add_mlflow_tag(reference_run_id, refinement_timestamp, suffix="reference")

    num_refinement_connections = _write_original_knowledge(refinement_config)
    refinement_run_id = _main()
    _add_mlflow_tag(refinement_run_id, refinement_timestamp, suffix="original")

    for i in range(refinement_config.num_refinements):
        num_new_refinement_connections = _write_refined_knowledge(
            refinement_config,
            reference_run_id=reference_run_id,
            refinement_run_id=refinement_run_id,
        )
        if num_refinement_connections == num_new_refinement_connections:
            logging.info(
                "Refined knowledge has same number of connections as previous knowledge, aborting refinement!"
            )
            return
        if num_reference_connections == num_new_refinement_connections:
            logging.info(
                "Refined knowledge has same number of connections as reference knowledge, aborting refinement!"
            )
            return

        refinement_run_id = _main()
        _add_mlflow_tag(
            refinement_run_id, refinement_timestamp, suffix="refinement_" + str(i)
        )
        num_refinement_connections = num_new_refinement_connections

    logging.info("Finished refinement run ({group})".format(group=refinement_timestamp))

def _get_knowledge_from_id(run_id: str):
    run_knowledge_path = Path(
            refinement_config.mlflow_dir
            + "{run_id}/artifacts/knowledge.json".format(run_id=run_id)
        )
    if not run_knowledge_path.exists():
        logging.debug("No knowledge file for run {} in MlFlow dir".format(run_id))
        return {}

    with open(run_knowledge_path) as f:
        return json.load(f)

def _do_original_for_generation(timestamp: int, config: refinement.RefinementConfig) -> Tuple[str, Dict[str, List[str]]]:
    if len(refinement_config.original_run_id) > 0:
        original_run_id = refinement_config.original_run_id
    else:
        # Do not use _write_original_knowledge, because we reuse the noise parameter for the generation 
        original_knowledge = refinement.KnowledgeProcessor(config).load_original_knowledge()
        _write_file_knowledge(original_knowledge)
        original_run_id = _main()
        _add_mlflow_tags_for_new_run(original_run_id, timestamp, "original")

    loaded_knowledge = _get_knowledge_from_id(original_run_id)
    return (original_run_id, loaded_knowledge)

def _get_noise_amount_for_iteration(config: refinement.RefinementConfig, i: int) -> float:
    return config.edges_to_add[min(i, len(config.edges_to_add) - 1)]

def main_generation(refinement_config: refinement.RefinementConfig):
    refinement_timestamp = time.time()

    generator = np.random.default_rng(ExperimentConfig().random_seed)

    (original_run_id, baseline_knowledge) = _do_original_for_generation(refinement_timestamp, refinement_config)
    refinement_run_ids = [original_run_id]
    num_connections_original = calculate_num_connections(baseline_knowledge)
    combined_knowledge = baseline_knowledge.copy()

    processor = refinement.KnowledgeProcessor(refinement_config)
    
    for i in range(refinement_config.num_refinements):
        noise_amount = _get_noise_amount_for_iteration(refinement_config, i)
        logging.info("Adding %f noise to current knowledge", noise_amount)
        generated_knowledge = _add_random_connections(combined_knowledge, noise_amount, generator)
        _write_file_knowledge(generated_knowledge)
        current_run_id = _main()
        _add_mlflow_tags_for_refinement(current_run_id, refinement_timestamp, 2 * i, refinement_config)
        refinement_run_ids.append(current_run_id)

        logging.info("Removing harmful generated edges...")
        refined_knowledge = processor.load_refined_knowledge(refinement_run_ids[-1], refinement_run_ids[-2])
        if refinement_config.refinement_window_size > -1:
            baseline_index = max(0, 2 * (i - refinement_config.refinement_window_size))
            baseline_knowledge = _get_knowledge_from_id(refinement_run_ids[baseline_index])
        
        # Ensure that only generated edges can be removed
        for child, parents in refined_knowledge.items():
            combined_knowledge[child] = list(set(baseline_knowledge.get(child, [])).union(set(parents)))

        num_connections_current = calculate_num_connections(combined_knowledge)
        if num_connections_current == num_connections_original:
            logging.info(
                "Refined knowledge has same number of connections as original knowledge, aborting refinement!"
            )
            break

        _write_file_knowledge(combined_knowledge)
        current_run_id = _main()
        _add_mlflow_tags_for_refinement(current_run_id, refinement_timestamp, 2 * i + 1, refinement_config)
        refinement_run_ids.append(current_run_id)

        logging.info(
            "Completed generation iteration {current} of {total}"
            .format(current=i+1, total=refinement_config.num_refinements)
        )

    logging.info("Finished generation run ({group})".format(group=refinement_timestamp))
    logging.info("original run id: {original_run_id}".format(original_run_id=original_run_id))

    for run in refinement_run_ids[1:]:
        logging.info("refinement run id: {refinement_run_id}".format(refinement_run_id=run))

if __name__ == "__main__":
    refinement_config = refinement.RefinementConfig()

    if refinement_config.mode == "v1":
        main_refinement(refinement_config)
    elif refinement_config.mode == "v2":
        main_boosting(refinement_config)
    elif refinement_config.mode == "gen":
        main_generation(refinement_config)
    elif refinement_config.mode == "inject":
        mlflow.set_experiment("Domain Guided Monitoring")
        refinement_timestamp = time.time()

        (original_run_id, state) = _do_original_run(refinement_timestamp, refinement_config)

        with mlflow.active_run() or mlflow.start_run() as run:
            current_run_id = run.info.run_id
            runner = ExperimentRunner()
            _log_all_configs_to_mlflow()

            with open("data/injected_attention.json", "r") as f:
                injected_attention = json.load(f)

            connections: Dict[Tuple[str, str], float] = {}
            for child, parents in injected_attention.items():
                for parent, score in parents.items():
                    # Scale percentage attention to score, which gets calculated to a percentage through the softmax
                    # Use 10000 to get approximate resolution of .01%

                    connections[(child, parent)] = float(score) * 10000

            state.model.embedding_layer.overwrite_attention_scores(connections)
            state.model.rnn_layer.trainable = refinement_config.freeze_rnn_sequence[0] != 1
            state.model.embedding_layer.trainable = refinement_config.freeze_embeddings_sequence[0] != 1
            state.model.activation_layer.trainable = refinement_config.freeze_activation_sequence[0] != 1

            _add_mlflow_tags_for_refinement(current_run_id, refinement_timestamp, 0, refinement_config)
            state = runner.run_from_state(current_run_id, state)

    else:
        logging.error("unknown refinement mode")
