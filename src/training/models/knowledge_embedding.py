from src.features.knowledge.base import BaseKnowledge
import tensorflow as tf
from tensorflow.python.training.tracking.data_structures import NoDependency
from typing import List, Dict, Tuple
from .config import ModelConfig
from .base import BaseEmbedding
import logging
from tqdm import tqdm


class KnowledgeEmbedding(BaseEmbedding, tf.keras.Model):
    def __init__(
        self, knowledge: BaseKnowledge, config: ModelConfig, embedding_name: str
    ):
        super(KnowledgeEmbedding, self).__init__()
        self.embedding_name = embedding_name
        self.config = config

        self.extended_vocab = knowledge.get_extended_vocab()
        self.num_features = len(knowledge.get_vocab())
        self.num_all_features = len(self.extended_vocab)
        self.num_hidden_features = self.num_all_features - self.num_features

        self.w = tf.keras.layers.Dense(
            self.config.attention_dim,
            use_bias=True,
            activation="tanh",
            kernel_regularizer=super()._get_kernel_regularizer(scope="attention"),
        )
        self.u = tf.keras.layers.Dense(
            1, use_bias=False, kernel_regularizer=super()._get_kernel_regularizer(scope="attention")
        )

        self._init_basic_embedding_variables(knowledge)
        self._init_connection_information(knowledge)
        self._init_corrective_terms()

    def _init_basic_embedding_variables(self, knowledge: BaseKnowledge):
        logging.info("Initializing %s basic embedding variables", self.embedding_name)
        self.basic_feature_embeddings = self.add_weight(
            initializer=self._get_feature_initializer(
                knowledge.get_description_vocab(set(knowledge.get_vocab().values()))
            ),
            trainable=self.config.base_feature_embeddings_trainable,
            name="{}/basic_feature_embeddings".format(self.embedding_name),
            shape=(self.num_features, self.config.embedding_dim),
            regularizer=super()._get_kernel_regularizer(scope="base_embeddings"),
        )
        self.basic_hidden_embeddings = self.add_weight(
            initializer=self._get_hidden_initializer(
                knowledge.get_description_vocab(set(knowledge.get_vocab().values()))
            ),
            trainable=self.config.base_hidden_embeddings_trainable,
            name="{}/basic_hidden_embeddings".format(self.embedding_name),
            shape=(self.num_hidden_features, self.config.embedding_dim),
            regularizer=super()._get_kernel_regularizer(scope="base_embeddings"),
        )

    def _init_connection_information(self, knowledge: BaseKnowledge):
        logging.info("Initializing %s connection information", self.embedding_name)
        self.connections: Dict[int, List[int]] = NoDependency({})

        self.partition_end = [0] * self.num_features

        for idx in range(self.num_features):
            connected_idxs = knowledge.get_connections_for_idx(idx)
            self.connections[idx] = sorted(list(connected_idxs))
            self.partition_end[idx] = self.partition_end[idx - 1] + len(connected_idxs)

        self.connection_partition: List[int] = [0] * self.partition_end[-1]
        # connection_partition[i] = j -> {connection i relevant for j}

        for idx in range(self.num_features):
            begin = self.partition_end[idx - 1] if idx > 0 else 0
            end = self.partition_end[idx]

            for i in range(begin, end):
                self.connection_partition[i] = idx

        # Sorted by insertion order by default since Python 3.7
        self.connection_indices = list(self.connections.values())
        # exists j such that connection_indices[i][j] = k -> feature i is connected to feature k

        self.flattened_connection_indices = [
            x for sublist in self.connection_indices for x in sublist
        ]  # connection k is between connection_partition[k] and flattened_connection_indices[k]
        # connection_indices[i,j] = k -> connection_partition[l]=i, flattened_connection_indices[l]=k
        self.num_connections = len(self.flattened_connection_indices)

    def _calculate_attention_distribution_lookups(self):
        scores = self._calculate_raw_scores()

        scores_per_outgoing_feature = tf.ragged.stack_dynamic_partitions(
            scores,
            partitions=self.connection_partition,
            num_partitions=self.num_features
        ) # shape: (num_features, num_connections per feature)

        scores_per_incoming_feature = tf.ragged.stack_dynamic_partitions(
            scores,
            partitions=self.flattened_connection_indices,
            num_partitions=self.num_all_features
        ) # shape: (num_all_features, num_connections per feature)

        return (self.connection_partition, self.flattened_connection_indices, scores, scores_per_outgoing_feature, scores_per_incoming_feature)

    def _init_corrective_terms(self):
        self.corrective_terms = self.add_weight(
            initializer=tf.ones_initializer(),
            trainable=False,
            name="{}/corrective_terms".format(self.embedding_name),
            shape=(self.num_connections),
        )

    def get_corrective_terms_as_list(self):
        return self.corrective_terms.numpy().tolist()

    def set_corrective_terms_from_list(self, term_list: List[float]):
        self.corrective_terms.assign(tf.convert_to_tensor(term_list, dtype=tf.float32))

    def update_corrective_terms(self, connections: Dict[Tuple[str, str], float]):
        # Ensure that connections that should not be corrected don't have any effect
        term_list = [1.0] * self.num_connections

        for ((outgoing, incoming), term) in connections.items():
            index = self._get_connection_index_from_labels(outgoing, incoming)
            term_list[index] = term

        new_corrective_terms = tf.convert_to_tensor(term_list, dtype=tf.float32)
        self.corrective_terms.assign(tf.multiply(self.corrective_terms, new_corrective_terms))

    def _get_connection_index_from_labels(self, outgoing: str, incoming: str):
        outgoing_idx = self.extended_vocab[outgoing]
        incoming_idx = self.extended_vocab[incoming]

        connection_offset = self.connections[outgoing_idx].index(incoming_idx)

        return (self.partition_end[outgoing_idx - 1] if outgoing_idx > 0 else 0) + connection_offset

    def _load_connection_embedding_matrix(self):
        embeddings = tf.concat(
            [self.basic_feature_embeddings, self.basic_hidden_embeddings],
            axis=0,
            name="all_feature_embeddings",
        )  # shape: (num_all_features, embedding_size)
        return tf.gather(
            embeddings,
            self.flattened_connection_indices,
            name="connected_embeddings_per_connection",
        )  # shape: (num_connections, embedding_size)

    def _load_attention_embedding_matrix(self):
        connection_embedding_matrix = self._load_connection_embedding_matrix()
        feature_embedding_matrix = tf.gather(
            self.basic_feature_embeddings,
            self.connection_partition,
            axis=0,
            name="feature_embeddings_per_connection",
        )  # shape: (num_connections, embedding_size)
        return tf.concat(
            [feature_embedding_matrix, connection_embedding_matrix],
            axis=1,
            name="concatenated_connection_embeddings",
        )  # (num_connections, 2*embedding_size)

    def _calculate_raw_scores(self):
        attention_embedding_matrix = self._load_attention_embedding_matrix()

        scores = self.u(
            self.w(attention_embedding_matrix)
        )  # shape: (num_connections, 1)
        scores = tf.math.exp(scores)

        if self.config.use_corrective_terms:
            scores = tf.multiply(scores, self.corrective_terms[:,tf.newaxis])
        
        return scores

    def _calculate_attention_embeddings(self):
        connection_embedding_matrix = self._load_connection_embedding_matrix()

        scores = self._calculate_raw_scores()

        scores_per_feature = tf.ragged.stack_dynamic_partitions(
            scores,
            partitions=self.connection_partition,
            num_partitions=self.num_features,
            name="attention_scores_per_feature",
        )  # shape: (num_features, num_connections per feature)
        score_sum_per_feature = tf.reduce_sum(
            scores_per_feature, axis=1, name="attention_score_sum_per_feature",
        )  # shape: (num_features, 1)
        attention_weights = scores_per_feature / tf.expand_dims(
            score_sum_per_feature,
            axis=1,
            name="normalised_attention_scores_per_feature",
        )  # shape: (num_features, num_connections per feature)

        connections_per_feature = tf.ragged.stack_dynamic_partitions(
            connection_embedding_matrix,
            partitions=self.connection_partition,
            num_partitions=self.num_features,
            name="connection_embeddings_per_feature",
        )  # shape: (num_features, num_connections per feature, embedding_size)
        context_vector = (
            attention_weights * connections_per_feature
        )  # shape: (num_features, num_connections, embedding_size)
        context_vector = tf.reduce_sum(
            context_vector, axis=1, name="context_vector",
        )  # shape: (num_features, embedding_size)

        return (context_vector, attention_weights)

    def _final_embedding_matrix(self):
        context_vector, _ = self._calculate_attention_embeddings()
        return context_vector

    def call(
        self, values
    ):  # values shape: (dataset_size, max_sequence_length, num_features)
        embedding_matrix = self._final_embedding_matrix()
        return tf.linalg.matmul(
            values, embedding_matrix,
        )  # shape: (dataset_size, max_sequence_length, embedding_size)
