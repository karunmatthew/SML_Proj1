import pandas as pd
import sys
from tqdm import tqdm
from gensim.models import Word2Vec
import random

READ = "r"
# stores the mapping from a person to the people they follow
follows = {}
unique_ids = set({})
follow_counts = {}

# --------- Hyper-parameters ----------- #
RANDOM_WALK_LENGTH = 10
WALKS_PER_NODE = 8000
# -------------------------------------- #

# TODO change the path
TRAINING_FILE_PATH = '/home/student.unimelb.edu.au/kvarghesemat/PycharmProjects/SML/train.txt'
MODEL_PATH = '/home/student.unimelb.edu.au/kvarghesemat/PycharmProjects/SML/rw.model'
WORD_VECTORS_PATH = '/home/student.unimelb.edu.au/kvarghesemat/PycharmProjects/SML/word_vectors'
WORD_VECTOR_MATRIX = "/home/student.unimelb.edu.au/kvarghesemat/PycharmProjects/SML/rw3"


# load the follows data-structure
def load_training_data():
    train_data = open(TRAINING_FILE_PATH, READ)
    for entry in train_data:
        ids = entry.strip().split('\t')
        # add all the entries to the SET to find the unique ids in the training data
        unique_ids.update(ids)

        # extract the source
        source = ids[0]
        # extract the people who the user follows, by removing the user-id from the ids
        sinks = set(ids[1:])

        # maps user to the people they follow
        follows[source] = sinks

        # keep a map of no of people they follow
        follow_counts[source] = len(sinks)


# performs a single random walk on the passed node for the given number of hops
# please note that this walk is ALWAYS from a source to a sink
def perform_random_walk(node):

    # stores the path of a single random walk
    random_walk = [node]
    hops = RANDOM_WALK_LENGTH
    while hops >= 0:
        hops -= 1
        if node in follows:
            neighbours = list(follows[node])
            # remove the neighbours already visited
            neighbours = list(set(neighbours) - set(random_walk))
            if len(neighbours) == 0:
                break
        else:
            # cannot go any further
            break

        # choose one of the neighbouring nodes
        random_node = random.choice(neighbours)
        random_walk.append(random_node)
        # jump to the new node
        node = random_node

    return random_walk


# returns an array of random walks
# they all are NOT of the same length
def get_random_walks():
    random_walks = []
    # perform random walks for all nodes who follow somebody
    for unique_node in tqdm(unique_nodes):
        if unique_node in follows:
            for index in range(min(WALKS_PER_NODE, follow_counts[unique_node])):
                random_walks.append(perform_random_walk(unique_node))

    return random_walks


def create_and_save_model():
    random_walks = get_random_walks()
    # count of sequences
    len(random_walks)
    # hs = 0 was what we gave previously
    # sg=1 means skip-gram
    # alpha : initial alpha 0.03
    # min_alpha : the learning rate to which the system drops to as things progress 0.0007
    # model = Word2Vec(alpha=0.03, window=4, sg=1, hs=0,
    #                 negative=10,  min_alpha=0.0007,
    #                 seed=99)
    model = Word2Vec(alpha=0.03, window=3, sg=1, hs=1,
                     negative=10, min_alpha=0.0005,
                     seed=99)
    model.build_vocab(random_walks)
    model.train(random_walks, total_examples=model.corpus_count, epochs=20)
    model.save(MODEL_PATH)
    model.init_sims(replace=True)

    model.wv.save_word2vec_format(WORD_VECTOR_MATRIX)
    model.wv.save(WORD_VECTORS_PATH)

    # perform some sanity checks
    print(model.similar_by_word('540762'))
    print(model.wv.most_similar(positive=["540762"]))
    print(model.wv.similarity("540762", '2345899'))


load_training_data()

# get the list of all unique nodes
unique_nodes = list(unique_ids)

create_and_save_model()


