import numpy as np

class Evaluation:

    def __init__(self, predicted, probabilities, expected, expected_triples, other_index, all_pairs, clusters, is_negative):
        assert \
            len(predicted) == len(probabilities) and len(predicted) == len(expected), \
            "All inputs should have same length but got {}, {}, {}.".format(len(predicted), len(probabilities),
                                                                            len(expected))

        self.predicted = predicted
        self.probabilities = probabilities
        self.expected = expected
        self.expected_triples = expected_triples
        self.other_index = other_index
        self.clusters = clusters
        self.all_pairs = all_pairs
        self.is_negative = is_negative
        self.grouped = {}

    # Hits@n metric based on the evaluation of Tim Dettmers (ConvE)
    # (https://github.com/TimDettmers/ConvE/blob/master/evaluation.py)
    def hits_n(self, n):

        hits = []
        for i in range(len(self.predicted)):
            is_hit = np.any(self.predicted[i][:n] == self.expected[i][:n])

            if is_hit:
                hits.append(1.0)
            else:
                hits.append(0.0)
        return np.mean(hits)

    # Hits metric with fixed threshold instead of fixed n
    def hits_threshold(self, threshold, verbose=1):
        hits = []
        for i in range(len(self.predicted)):
            is_hit = np.any(self.probabilities[i][np.nonzero(self.predicted[i] == self.expected[i])] > threshold)
            if verbose > 1:
                print(i, is_hit)
            if is_hit:
                hits.append(1.0)
            else:
                hits.append(0.0)

        if verbose > 0:
            print(np.unique(hits, return_counts=True))
        return np.mean(hits)

    # Hits metric taking negative examples based on clustering into account
    def hits_neg_threshold(self, threshold, verbose=1):
        counts = []
        hits = []
        for i in range(len(self.predicted)):
            guess_indices = np.nonzero(self.probabilities[i] > threshold)
            guesses = self.predicted[i][guess_indices]
            is_hit = False
            has_miss = False

            # log the number of predictions
            counts.append(len(guesses))

            for g in guesses:
                if g == self.expected[i]:
                    is_hit = True
                if self.is_negative(g, self.expected_triples[i][self.other_index[i]], self.cluster, self.all_pairs):
                    has_miss = True
                if has_miss and is_hit:
                    break

            if verbose > 1:
                print(i, self.expected_triples[i], 'hit:', is_hit, 'miss:', has_miss)

            if is_hit:
                hits.append(1.0)
            else:
                hits.append(0.0)

            if has_miss:
                hits.append(0.0)
            else:
                hits.append(1.0)

        if verbose > 0:
            print(np.unique(hits, return_counts=True))

        return np.mean(hits), counts

    def hits_group(self, threshold):

        if len(self.grouped) == 0:
            print("Grouping data...")
            # FastText preprocesses the data so that every second triple has the head as label and the rest the tail respectively
            tails = list(zip(self.expected_triples[1::2, :2].tolist(), self.expected_triples[1::2, 2:].tolist()))
            heads = list(zip(self.expected_triples[::2, -1:0:-1].tolist(), self.expected_triples[::2, :1].tolist()))

            x_tails, y_tails, keys = self._cluster_test_data(tails, self.predicted[1::2], self.probabilities[1::2])
            x_heads, y_heads, _ = self._cluster_test_data(heads, self.predicted[::2], self.probabilities[::2])

            self.grouped['x'] = {**x_heads, **x_tails}
            self.grouped['y'] = {**y_heads, **y_tails}

        return self._hits_batch_threshold(self.grouped['x'], self.grouped['y'], threshold)

    @staticmethod
    def _cluster_test_data(expected, predictions, probabilities):
        x = {}
        y = {}
        y_indices = {}

        i = 0
        for e in expected:
            key = (e[0][0], e[0][1])
            y_key = i

            if key not in x:
                x[key] = []
                y[key] = (predictions[y_key], probabilities[y_key])
                y_indices[key] = []

            x[key].append(e[1][0])
            y_indices[key].append(y_key)
            i += 1

        return x, y, y_indices

    # Hits metric that groups the data before evaluation and considers negative samples
    def _hits_batch_threshold(self, x, y, threshold, verbose=True):
        assert \
            len(y) == len(x), \
            "x and y should have same length but were {}, {}.".format(len(x), len(y))

        counts = []
        hits = []
        for k in x.keys():
            guess_indices = np.nonzero(y[k][1] > threshold)
            guesses = y[k][0][guess_indices]

            # log the number of predictions
            counts.append(len(guesses))

            for e in x[k]:
                if e in guesses:
                    hits.append(1.0)
                else:
                    hits.append(0.0)

            for g in guesses:
                if self.is_negative(g, k[0], self.cluster, self.all_pairs):
                    hits.append(0.0)

        if verbose:
            print(np.unique(hits, return_counts=True))
        return np.mean(hits), counts