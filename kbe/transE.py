from .embedder import *
import tensorflow as tf


class TransE(AbstractEmbedder):

    def score(self, head, relation, tail):
        head_embedding = tf.nn.embedding_lookup(self._embedding, head)
        relation_embedding = tf.nn.embedding_lookup(self._embedding, self.node_vocab_size + relation)
        tail_embedding = tf.nn.embedding_lookup(self._embedding, tail)

        return tf.norm(head_embedding + relation_embedding - tail_embedding)

    def _initialize_embedding(self):
        self._embedding = tf.Variable(
            tf.random_normal([self.node_vocab_size + self.relation_vocab_size, self.embedding_size], stddev=0.1),
            name="embedding")

    @property
    def embedding(self):
        return self.session.run(self._embedding)