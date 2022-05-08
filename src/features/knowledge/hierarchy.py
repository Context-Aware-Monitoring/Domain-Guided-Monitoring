import pandas as pd
from typing import Dict, Optional, Set
from tqdm import tqdm
import logging
from .node import Node
from .base import BaseKnowledge
from .config import KnowledgeConfig


class HierarchyKnowledge(BaseKnowledge):
    def __init__(
        self,
        config: KnowledgeConfig,
        child_id_col="child_id",
        parent_id_col="parent_id",
        child_name_col="child_name",
        parent_name_col="parent_name",
    ):
        super(HierarchyKnowledge, self).__init__(config=config)
        self.child_id_col = child_id_col
        self.parent_id_col = parent_id_col
        self.child_name_col = child_name_col
        self.parent_name_col = parent_name_col

    def get_connections_for_idx(self, idx: int) -> Set[int]:
        return set(self.nodes[idx].get_ancestor_label_idxs() + [idx])

    def is_connected(self, child: str, parent: str) -> bool:
        if self.mapping_table is None:
            return super().is_connected(child, parent)

        return (parent, child) in self.mapping_table

    def get_description_vocab(self, ids: Set[int]) -> Dict[int, str]:
        return {idx: node.label_name for idx, node in self.nodes.items() if idx in ids}

    def _build_mapping_table(
        self, hierarchy_df: pd.DataFrame,
        hierarchy_mapping_df: pd.DataFrame,
        input_level: str, output_level: str
    ):
        lookup = hierarchy_mapping_df.set_index(input_level).to_dict()
        lookup = lookup[output_level]

        # Ensure root node remains unchanged
        lookup["-1"] = "-1"

        make_format = lambda l, x: "{level}#{value}".format(level=l,value=x)
        lookup = {make_format(input_level, k): make_format(output_level, v) for k, v in lookup.items()}

        map_df = hierarchy_df.copy(deep=True)
        map_df[self.parent_id_col] = (
            map_df[self.parent_id_col].replace(to_replace=lookup)
        )

        return set(map_df
                .reindex(columns=[self.parent_id_col, self.child_id_col])
                .drop_duplicates()
                .to_records(index=False)
                .tolist()
        )

    def build_hierarchy_from_df(
        self, hierarchy_df: pd.DataFrame, vocab: Dict[str, int],
        hierarchy_mapping_df: Optional[pd.DataFrame] = None,
        input_level: str = "",
        output_level: str = ""
    ):
        self.vocab: Dict[str, int] = vocab

        if hierarchy_mapping_df is not None:
            self.mapping_table = self._build_mapping_table(
                hierarchy_df, hierarchy_mapping_df,
                input_level, output_level
            )
        else:
            self.mapping_table = None

        self._build_extended_vocab(hierarchy_df, vocab)
        for _, row in tqdm(hierarchy_df.iterrows(), desc="Building Hierarchy from df"):
            child_id = row[self.child_id_col]
            if child_id not in self.extended_vocab:
                logging.debug("Ignoring node %s as not in dataset", child_id)
                continue

            child_node = self.nodes[self.extended_vocab[child_id]]
            parent_node = self.nodes[self.extended_vocab[row[self.parent_id_col]]]

            if child_node is not parent_node:
                child_node.in_nodes.add(parent_node)
                parent_node.out_nodes.add(child_node)

        logging.info("Built hierarchy with %d nodes", len(self.nodes))

    def _build_extended_vocab(self, hierarchy_df: pd.DataFrame, vocab: Dict[str, int]):
        self.extended_vocab: Dict[str, int] = {}
        self.nodes: Dict[int, Node] = {}

        labels_to_handle = sorted(vocab.keys())
        max_index = max(vocab.values())
        while len(labels_to_handle) > 0:
            label = labels_to_handle.pop()
            if label in self.extended_vocab:
                continue

            if label in vocab:
                self.extended_vocab[label] = vocab[label]
            else:
                self.extended_vocab[label] = max_index + 1
                max_index = max_index + 1

            parents_df = hierarchy_df[hierarchy_df[self.child_id_col] == label]
            label_names = set(parents_df[self.child_name_col])
            label_names.update(
                set(
                    hierarchy_df[hierarchy_df[self.parent_id_col] == label][
                        self.parent_name_col
                    ]
                )
            )
            self.nodes[self.extended_vocab[label]] = Node(
                label_idx=self.extended_vocab[label],
                label_str=label,
                label_names=label_names,
            )

            parents = sorted(set(parents_df[self.parent_id_col]))
            labels_to_handle = labels_to_handle + parents

        self.extra_vocab: Dict[str, int] = {
            k: v for k, v in self.extended_vocab.items() if k not in self.vocab
        }

    def __str__(self):
        roots = [node for node in self.nodes.values() if node.is_root()]
        all_strings = []
        for root in roots:
            all_strings = all_strings + self._to_string_recursive(root, "")
        return "\n".join(all_strings)

    def _to_string_recursive(self, current_node, current_prefix):
        strings = [current_prefix + current_node.label_str]
        for node in current_node.out_nodes:
            strings = strings + self._to_string_recursive(node, current_prefix + "-")

        return strings
