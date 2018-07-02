import tensorflow as tf
import random


class AbstractEmbedder:

    def __init__(self, margin, batch_size, epochs, learning_rate, embedding_size):
        """
        This is an abstract superclass for embedding methods. Embedders embed the nodes and edges of a knowledge graph
        so that they can be used to predict missing parts of the triples.

        :param margin: The maring in the loss function.
        :param batch_size: The batch size during training.
        :param epochs: The number of epochs the training should go through.
        :param learning_rate: This is how fast the optimizer should descent during training.
        :param embedding_size: The size of the embeddings created. Depending on the method this is a scalar or a tuple.
        :param n: The number of
        """
        self.session = tf.Session()
        self.margin = margin
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self._initialize_embedding()

    def _initialize_embedding(self, node_vocab_size, relation_vocab_size):
        self.node_vocab_size = node_vocab_size
        self.relation_vocab_size = relation_vocab_size

    def score(self, head, relation, tail):
        """
        The score function of the model.

        :param head: The head node represented as an integer.
        :param relation: The relation represented as an integer respectively.
        :param tail: The tail node represented like the head.
        :return: The score of this triple, usually something like the likelihood
                 that this triple is contained in the knowledge base
        """
        raise NotImplementedError()

    def loss(self, batch):
        """
        The loss function that is minimized during training.

        :param batch: The current training batch. Consists out of 5-tuples: (head, tail, relation, head_modified, tail_modified)
        :return: The loss value of this batch.
        """
        return tf.reduce_sum(self._loss_part(batch[0], batch[1], batch[2], batch[3], batch[4]))

    def _loss_part(self, head, tail, relation, head_modified, tail_modified):
        """
        Private method that is used to calculate the loss.
        This how the loss of a single element of the batch is calculated.

        All parameters are represented as their integer id.
        The modified parts are the generated negative examples.
        """
        return tf.max(0,
                      self.score(head, relation, tail)
                      + self.margin
                      - self.score(head_modified, relation, tail_modified))

    def train(self, triples, node_vocab_size=None, relation_vocab_size=None):
        """
        Train the embedding.
        :param triples: All training triples of format (head, relation, tail) where each is represented as an integer id
                        which should be between 0 (inclusive) and vocabulary size (exclusive).
        :param node_vocab_size: The total number of nodes. If None, the max value is used.
        :param relation_vocab_size: The total number of nodes. If None, the max value is used.
        :return: Nothing.
        """
        self._initialize_embedding(node_vocab_size, relation_vocab_size)
        self.session.run(tf.local_variables_initializer())
        for batch in tf.train.batch(triples, batch_size=self.batch_size, allow_smaller_final_batch=True):
            final_batch = []
            for element in batch:
                final_batch.append(element.appendAll(self._sample_corrupted_triple(element)))
            self.session.run(self._optimizer().minimize(self.loss(final_batch)))

    def _sample_corrupted_triple(self, head, relation, tail):
        """
        In classic methods either head or tail is corrupted which one is determined by random (50/50 chance).
        The method might be different depending on the model.
        :return: A corrupted triple for training.
        """
        if random.random() >= 0.5:
            return random.randint(0, self.node_vocab_size - 1), relation, tail
        else:
            return head, relation, random.randint(0, self.node_vocab_size - 1)

    def _optimizer(self):
        """
        Return the optimizer for training.
        :return: A subclass of tf.train.Optimizer
        """
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def evaluate(self, triples):

        pass

    def predict_tail(self, head, relation, n=10):
        """
        Predict the tail node of this triple.

        :param head: The head node as an integer.
        :param relation: The relation as an integer.
        :param n: Number of results.
        :return: The n top tail nodes in the format. You can use score() to calculate the distance.
        """
        raise NotImplementedError()

    def predict_relation(selfs, head, tail, n=10):
        """
        Predict the relation of this triple.

        :param head: The head node as an integer.
        :param tail: The tail node as an integer.
        :param n: Number of results.
        :return: The n top relations in the format. You can use score() to calculate the distance.
        """
        raise NotImplementedError()

    def predict_head(selfs, relation, tail, n=10):
        """
        Predict the head node of this triple.

        :param relation: The relation as an integer.
        :param tail: The tail node as an integer.
        :param n: Number of results.
        :return: The n top head nodes in the format. You can use score() to calculate the distance.
        """
        raise NotImplementedError()
