"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import Counter

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    LAPLACE_EMIT = 0.00001
    LAPLACE_TRANS = 0.00001

    cnt_emit = {}
    cnt_trans = {}
    emission = {}
    emission_UNK = {}
    transition = {}

    # Count occurrences of tags, tag pairs, tag/word pairs
    for i in train:
        for j in range(len(i)):
            if i[j][1] not in cnt_emit:
                cnt_emit[i[j][1]] = Counter()
            cnt_emit[i[j][1]][i[j][0]] += 1

            if j < len(i)-1:
                if i[j][1] not in cnt_trans:
                    cnt_trans[i[j][1]] = Counter()
                cnt_trans[i[j][1]][i[j+1][1]] += 1

    # Compute smoothed probabilities & Take the log of each probability
    for i in cnt_emit:
        emission[i] = {}

        n = sum(cnt_emit[i].values())
        V = len(list(cnt_emit[i]))
        for (k, v) in cnt_emit[i].items():
            emission[i][k] = math.log(v+LAPLACE_EMIT) - math.log(n+LAPLACE_EMIT*(V+1))
        emission_UNK[i] = math.log(LAPLACE_EMIT) - math.log(n+LAPLACE_EMIT*(V+1))

    tags = list(cnt_emit.keys())
    for i in cnt_trans:
        transition[i] = {}

        n = sum(cnt_trans[i].values())
        V = len(list(cnt_trans[i]))
        for j in tags:
            if j in cnt_trans[i]:
                transition[i][j] = math.log(cnt_trans[i][j]+LAPLACE_TRANS) - math.log(n+LAPLACE_TRANS*(V+1))
            else:
                transition[i][j] = math.log(LAPLACE_TRANS) - math.log(n+LAPLACE_TRANS*(V+1))
    transition['END'] = {}
 

    prediction = []
    for i in range(len(test)):
        prediction.append([])
        # Construct the trellis
        trellis = []
        for j in range(len(test[i])):
            trellis.append({})

            if j == 0:
                trellis[j]["START"] = (0, None)
            else:
                for (k1, v1) in trellis[j-1].items():
                    for (k2, v2) in transition[k1].items():
                        p = v1[0] + v2 + emission[k2].get(test[i][j], emission_UNK[k2])
                        if k2 not in trellis[j] or p > trellis[j][k2][0]:
                            trellis[j][k2] = (p, k1)

        # Return the best path through the trellis by backtracing
        tag = "END"
        max = None
        for (k, v) in trellis[len(test[i])-1].items():
            if max == None or max < v[0]:
                tag = k
                max = v[0]

        for j in range(len(test[i])-1, -1, -1):
            prediction[i].insert(0, (test[i][j], tag))
            tag = trellis[j][tag][1]
    
    return prediction