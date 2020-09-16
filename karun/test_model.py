from gensim.models import Word2Vec
import sys

TRAINING_FILE_PATH = '/home/student.unimelb.edu.au/kvarghesemat/PycharmProjects/SML/train.txt'
TESTING_FILE_PATH = '/home/student.unimelb.edu.au/kvarghesemat/PycharmProjects/SML/test-public.txt'
READ = "r"
WRITE = "w"
OUT_FILE_PATH = '/home/student.unimelb.edu.au/kvarghesemat/PycharmProjects/SML/sample.csv'

outfile = open(OUT_FILE_PATH, WRITE)

# Load the word2vec model
model = Word2Vec.load("/home/student.unimelb.edu.au/kvarghesemat/PycharmProjects/SML/rw.model")
train_data = open(TRAINING_FILE_PATH, READ)

# None marker, indicates that processing did not happen
NULL = -5

outfile.write('Id,Predicted' + '\n')

sink2sources = {}
source2sinks = {}
source2sink_counts = {}


# compares the similarity of source,
def get_source_as_sink_similarity(source, sink):
    connected_sinks = source2sinks[source]

    c_sinks = 0
    total_sim = 0

    for connected_sink in connected_sinks:
        if connected_sink in model.wv.vocab:
            c_sinks += 1
            # print('-----', model.wv.similarity(connected_sink, sink))
            total_sim = total_sim + model.wv.similarity(connected_sink, sink)

    # print(c_sinks)

    if c_sinks > 0:
        return total_sim / c_sinks
    else:
        return -5


def get_sink_as_source_similarity(source, sink, index):
    connected_sources = sink2sources[sink]

    c_sources = 0
    total_sim = 0

    # get the 10-most nearest neighbours to the source
    similar_nodes = model.wv.most_similar(positive=[source], topn=200)
    links_to_sink = 0
    scores = []
    for similar_source, similarity_score in similar_nodes:
        if similar_source in source2sinks and sink in source2sinks[similar_source]:
            links_to_sink += 1
            scores.append(similarity_score)

    # max_sim = -1
    # for connected_source in connected_sources:
    #     if connected_source in model.wv.vocab:
    #         temp_score = model.wv.similarity(source, connected_source)
    #         if temp_score > max_sim:
    #             max_sim = temp_score

    for connected_source in connected_sources:
        if connected_source in model.wv.vocab:
            c_sources += 1
            total_sim = total_sim + model.wv.similarity(connected_source, source)

    # print(c_sinks)

    if c_sources > 0:
        if links_to_sink >= 1:
            # print('\n')
            # print('Sink Degree ::', len(sink2sources[sink]))
            # print(index, ' :: ', source, ' :: ', sink)
            # print(links_to_sink, ":::", scores)
            # print((total_sim / c_sources))
            return max(scores)
        # return max_sim
        else:
            return total_sim / c_sources
    else:
        return NULL


# returns a representation for source
def get_representation_for_source(src):
    if src in model.wv.vocab:
        word_vector_rep = model.wv[src]
        # print(word_vector_rep)
    elif src in source2sinks and len(source2sinks[src]) > 0:
        # if the source is missing, but the source has some sinks in the train set
        # we then say that a source is defined by the avg of the sources which follows its sinks
        # need to weigh this in future !
        print('No Match', src)
        print(len(source2sinks[src]))

        connected_sinks = source2sinks[src]
        c_sinks = 0
        for sink in connected_sinks:
            if sink in model.wv.vocab:
                c_sinks += 1

        print(c_sinks)
    else:
        print('NOTHING', src)


# returns a rep for sink
def get_representation_for_sink(snk):
    if snk in model.wv.vocab:
        word_vector_rep = model.wv[snk]
        # print(word_vector_rep)
    else:
        print('No word vector', snk)


# loads the source to sinks and sink to sources map
def load_mappings():
    for entry in train_data:
        ids = entry.strip().split('\t')
        source = ids[0].strip()
        sinks = set(ids[1:])
        source2sinks[source] = sinks
        source2sink_counts[source] = len(sinks)
        for sink in sinks:
            if sink in sink2sources:
                sink2sources[sink.strip()].append(source)
            else:
                sink2sources[sink.strip()] = [source]


def predict_with_test_data():
    test_data = open(TESTING_FILE_PATH, READ)
    test_data.readline()
    no_match = 0

    index = 0

    for line in test_data:
        index += 1
        data = line.split('\t')
        source, sink = data[1].strip(), data[2].strip()

        direct_sim = NULL
        source_as_sink_sim = NULL
        sink_as_source_sim = NULL
        total = 1000002

        # if both source and sink have word vector representations
        if source in model.wv.vocab and sink in model.wv.vocab:
            direct_sim = model.wv.similarity(source, sink)

        # transform source as a fn of the sinks it follows and then calculate the similarity between the two
        if sink in model.wv.vocab:
            source_as_sink_sim = get_source_as_sink_similarity(source, sink)

        if source in model.wv.vocab:
            sink_as_source_sim = get_sink_as_source_similarity(source, sink, index)

        if direct_sim == NULL and source_as_sink_sim == NULL and sink_as_source_sim == NULL:
            no_match += 1

        if direct_sim == NULL:
            total = total - 1
        if source_as_sink_sim == NULL:
            total = total - 1
        if sink_as_source_sim == NULL:
            total = total - 1000000

        # total_probability = max(0, direct_sim) + max(0, source_as_sink_sim) + max(0, sink_as_source_sim) * 1000000

        total_probability = max(0, sink_as_source_sim)

        if total == 0:
            total_probability = 0.1
        # else:
        #    total_probability = total_probability / total

        # print(direct_sim, ' : ', source_as_sink_sim, ' : ', sink_as_source_sim, ' : ', total_probability)
        outfile.write(str(index) + ',' + str(total_probability) + '\n')

    print('No Matches :', no_match)


load_mappings()
predict_with_test_data()
outfile.close()
