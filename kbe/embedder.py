import tensorflow as tf
import random


class AbstractEmbedder:

    def __init__(self, margin, batch_size, epochs, learning_rate, embedding_size, node_vocab_size, relation_vocab_size):
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

        self.margin = margin
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.node_vocab_size = node_vocab_size
        self.relation_vocab_size = relation_vocab_size
        self.optimizer = None

        self.session = tf.Session()
        self._initialize_embedding()
        self._initialize_session()

    def _initialize_session(self):
        self.session.run(tf.global_variables_initializer())

    def _initialize_embedding(self, node_vocab_size, relation_vocab_size):
        raise NotImplementedError()

    @property
    def embedding(self):
        raise NotImplementedError()


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
        return tf.reduce_sum(self._loss_part(batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]))

    def _loss_part(self, head, tail, relation, head_modified, tail_modified):
        """
        Private method that is used to calculate the loss.
        This how the loss of a single element of the batch is calculated.

        All parameters are represented as their integer id.
        The modified parts are the generated negative examples.
        """
        return tf.maximum(0.,
                          tf.subtract(
                              tf.add(self.score(head, relation, tail),
                                     self.margin),
                              self.score(head_modified, relation, tail_modified)))

    def train(self, triples):
        """
        Train the embedding.
        :param triples: All training triples of format (head, relation, tail) where each is represented as an integer id
                        which should be between 0 (inclusive) and vocabulary size (exclusive).
        :param node_vocab_size: The total number of nodes. If None, the max value is used.
        :param relation_vocab_size: The total number of nodes. If None, the max value is used.
        :return: Nothing.
        """
        dataset = tf.data.Dataset.from_tensors(self._sample_corrupted_triple(triples))
        dataset.batch(self.batch_size).repeat(self.epochs)
        data_iter = dataset.make_initializable_iterator()
        minimize_op = self._optimizer().minimize(self.loss(data_iter.get_next()))

        self.session.run(data_iter.initializer)
        self._initialize_session() # required as the optimizer defines new variables
        return self.session.run(minimize_op)

    def _sample_corrupted_triple(self, data):
        """
        In classic methods either head or tail is corrupted which one is determined by random (50/50 chance).
        The method might be different depending on the model.
        :return: A corrupted triple for training.
        """
        shuffled_data = tf.random_shuffle(data)

        half = tf.cast(tf.round(tf.shape(data)[0] / 2), tf.int32)

        upper_batch = tf.concat([shuffled_data[:half],
                                 tf.reshape(shuffled_data[:half, 0], [-1, 1]),
                                 tf.random_uniform([half, 1], minval=0, maxval=self.node_vocab_size - 1, dtype=tf.int32)],
                                1)
        lower_batch = tf.concat([shuffled_data[half:],
                                 tf.random_uniform([half, 1], minval=0, maxval=self.node_vocab_size - 1, dtype=tf.int32),
                                 tf.reshape(shuffled_data[half:, 2], [-1, 1])
                                 ],
                                1)

        return tf.concat([upper_batch, lower_batch], 0)

    def _optimizer(self):
        """
        Return the optimizer for training.
        :return: A subclass of tf.train.Optimizer
        """
        if self.optimizer is None:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name="embedder/optimizer")
        return self.optimizer

    def evaluate(self, triples):
        """
        Evaluate the embedding.
        :param triples:
        :return:
        """

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
