import numpy as np

def read_prediction(path, prefix="__label__"):
    predictions = []
    probabilities = []

    file = open(path, 'r')
    for line in file:
        data = line.strip("\n").split(" ")
        cleaned = []

        for label in data[::2]:
            cleaned.append(label[len(prefix):])

        predictions.append(cleaned)
        probabilities.append(data[1::2])

    return np.array(predictions), np.array(probabilities).astype(np.float64)


def read_clusters(path):
    cluster = {}

    file = open(path, 'r')
    for line in file:
        data = line.strip("\n").split(" ")
        cluster[data[0]] = int(data[1])

    return cluster


def read_expected(path, prefix="__label__"):
    expected = []
    triples = []
    other_index = []

    file = open(path, 'r')
    for line in file:
        data = line.strip("\n").split()

        i = 0
        i_label = 0
        triple = []
        for d in data:
            if d[:len(prefix)] == prefix:
                label = d[len(prefix):]
                triple.append(label)
                i_label = i
            else:
                triple.append(d)
            i += 1

        expected.append(label)
        triples.append(triple)

        if i_label == 0:
            other_index.append(2)
        else:
            other_index.append(0)

    return np.array(expected), np.array(triples), np.array(other_index)


def read_all_pairs(path):
    all_pairs = set()

    file = open(path, 'r')
    for line in file:
        data = line.strip("\n").split()
        pair = (data[0], data[1])

        all_pairs.add(pair)
    return all_pairs


def read_dataset(name = 'fb15k'):
    datasets = {'fb15k': {'path': 'fb15k', 'expected': 'data/ft_freebase_mtr100_mte100-test.txt'},
                'wn': {'path': 'WN', 'expected': 'data/ft_wordnet-mlj12-test.txt'}}
    dataset = datasets[name]

    all_pairs = read_all_pairs("data/{}/all.txt".format(dataset['path']))
    cluster = read_clusters("data/{}/clusters.txt".format(dataset['path']))
    y_pred, y_prob = read_prediction("data/{}/predictions.txt".format(dataset['path']))
    y_exp, triples_exp, other_exp = read_expected(dataset['expected'])

    return y_pred, y_prob, y_exp, triples_exp, other_exp, all_pairs, cluster