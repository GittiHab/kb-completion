import unittest
from ..transE import TransE

class EmbedderTest(unittest.TestCase):

    def setUp(self):
        self.embedder = TransE(1, 1, 3, 0.1, 3)

    def test_executable(self):
        triples = [(1, 1, 0),(1, 0, 2),(0, 0, 1),(2, 0, 2)]
        self.embedder.train(triples, 3, 3)