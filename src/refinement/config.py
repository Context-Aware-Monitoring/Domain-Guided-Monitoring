import dataclass_cli
import dataclasses
from pathlib import Path
from typing import List


@dataclass_cli.add
@dataclasses.dataclass
class RefinementConfig:
    num_refinements: int = 1
    min_edge_weight: float = 0.8
    max_train_examples: int = 10
    refinement_metric: str = "mean_outlier_score"
    refinement_metric_maxrank: int = -1
    max_edges_to_remove: int = 10
    max_refinement_metric: int = -1
    original_file_knowledge: Path = Path("data/original_file_knowledge.json")
    edges_to_add: float = -1
    reference_file_knowledge: Path = Path("data/reference_file_knowledge.json")
    mlflow_dir: str = "mlruns/1/"
    corrective_factor: float = 0.2
    original_run_id: str = ""
    reference_run_id: str = ""
    keep_state_from_original: bool = False
    freeze_rnn_sequence: List[int] = dataclasses.field(
        default_factory=lambda: [1] # Use int because argparse doesn't handle bool correctly
    )
    freeze_embeddings_sequence: List[int] = dataclasses.field(
        default_factory=lambda: [1]
    )
    freeze_activation_sequence: List[int] = dataclasses.field(
        default_factory=lambda: [1]
    )