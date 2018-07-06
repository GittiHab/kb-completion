import unittest
from ..transE import TransE

class EmbedderTest(unittest.TestCase):

    def setUp(self):
        self.embedder = TransE(1, 1, 3, 0.1, 3, 3, 3)
        self.triples = [(1, 1, 0),(1, 0, 2),(0, 0, 1),(2, 0, 2)]

    def test_executable(self):
        self.embedder.train(self.triples)

    def test_sanity(self):
        previous = self.embedder.embedding
        self.embedder.train(self.triples)
        self.assertFalse((previous==self.embedder.embedding).all(), "Embedding matrix should not stay equal after training")
