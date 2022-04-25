import logging
from src import _main
from src import features, refinement
import json
import time
from typing import Dict, List
from mlflow.tracking import MlflowClient
import random
from src.main import _log_all_configs_to_mlflow
from src import ExperimentRunner
import mlflow

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

def _add_random_connections(original_knowledge: Dict[str, List[str]], percentage: float = 0.1) -> Dict[str, List[str]]:
    connections = set([(child, parent) for child, parents in original_knowledge.items() for parent in parents if child != parent])
    children = list(set([x[0] for x in connections]))
    parents = list(set([x[1] for x in connections]))
    potential_connections = [
        (c, p) for c in children for p in parents if (c,p) not in connections and c != p
    ]
    connections_to_add = random.sample(
        potential_connections,
        k=min(len(potential_connections), int(percentage * len(connections)))
    )

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
    if refinement_config.edges_to_add > 0:
        logging.info("Adding %f noise to original knowledge", refinement_config.edges_to_add)
        original_knowledge = _add_random_connections(original_knowledge, refinement_config.edges_to_add)
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


def _do_reference_run(timestamp: int, config: refinement.RefinementConfig):
    if len(config.reference_run_id) > 0:
        return config.reference_run_id
    else:
        _write_reference_knowledge(config)
        reference_run_id = _main()
        _add_mlflow_tags_for_new_run(reference_run_id, timestamp, "reference")
        return reference_run_id


def main_boosting():
    mlflow.set_experiment("Domain Guided Monitoring")

    refinement_timestamp = time.time()
    refinement_config = refinement.RefinementConfig()

    reference_run_id = _do_reference_run(refinement_timestamp, refinement_config)

    refinement_run_ids = []

    with mlflow.start_run() as run:
        _log_all_configs_to_mlflow()
        runner = ExperimentRunner(run.info.run_id)
        runner.config.model_type = "gram" # Manually overwrite to get proper knowledge
        experiment_run = runner.run() # Original run as baseline
        refinement_run_id = run.info.run_id

    _add_mlflow_tags_for_new_run(refinement_run_id, refinement_timestamp, "original")
    original_run_id = refinement_run_id
    refinement_run_ids = []

    for i in range(refinement_config.num_refinements):
        refinement.KnowledgeProcessor(refinement_config).update_corrective_terms(experiment_run, refinement_run_id, reference_run_id)
        experiment_run.model.prediction_model.trainable = False

        with mlflow.start_run() as run:
            _log_all_configs_to_mlflow()
            runner = ExperimentRunner(run.info.run_id)
            runner.config.n_epochs = 1 # Lower epoch because weights are frozen anyway
            experiment_run = runner.continue_run(experiment_run)
            refinement_run_id = run.info.run_id
            _add_mlflow_tags_for_refinement(refinement_run_id, refinement_timestamp, i, refinement_config)
            refinement_run_ids = refinement_run_ids + [refinement_run_id]

    logging.info("Finished boosting run")
    logging.info("reference run id: {reference_run_id}".format(reference_run_id=reference_run_id))
    logging.info("original run id: {original_run_id}".format(original_run_id=original_run_id))

    for run in refinement_run_ids:
        logging.info("refinement run id: {refinement_run_id}".format(refinement_run_id=run))

def main_refinement():
    refinement_timestamp = time.time()
    refinement_config = refinement.RefinementConfig()
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

if __name__ == "__main__":
    #main_refinement()
    main_boosting()
