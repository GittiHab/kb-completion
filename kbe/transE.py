from .embedder import *
import tensorflow as tf


class TransE(AbstractEmbedder):

    def score(self, head, relation, tail):
        head_embedding = tf.nn.embedding_lookup(self.embedding, head)
        relation_embedding = tf.nn.embedding_lookup(self.node_vocab_size + self.embedding, relation)
        tail_embedding = tf.nn.embedding_lookup(self.embedding, tail)

        return tf.nn.l2_normalize(head_embedding + relation_embedding - tail_embedding)

    def _initializeEmbedding(self, node_vocab_size, relation_vocab_size):
        self.node_vocab_size = node_vocab_size
        self.relation_vocab_size = relation_vocab_size
        self.embedding_size = tf.Variable(tf.random_normal(
                [self.embedding_size, node_vocab_size + relation_vocab_size],
                stddev=0.1),
            name="embedding")
