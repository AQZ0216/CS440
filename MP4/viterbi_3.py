"""
Part 4: Here should be your best version of viterbi,
with enhancements such as dealing with suffixes/prefixes separately
"""

import math
from collections import Counter
import re

def viterbi_3(train, test):
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
    emission_LY = {}
    emission_ING = {}
    emission_ED = {}
    emission_ER = {}
    emission_EST = {}
    emission_TIVE = {}
    emission_NUM = {}
    emission_S = {}
    transition = {}

    dict_word = {}

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

            if i[j][0] in dict_word:
                dict_word[i[j][0]] = (False, None)
            else:
                dict_word[i[j][0]] = (True, i[j][1])

    hapax = {}
    X_LY = {}
    X_ING = {}
    X_ED = {}
    X_ER = {}
    X_EST = {}
    X_TIVE = {}
    X_NUM = {}
    X_S = {}

    for (k, v) in dict_word.items():
        if v[0]:
            if k.endswith("ly"):
                if v[1] in X_LY:
                    X_LY[v[1]] += 1
                else:
                    X_LY[v[1]] = 1
            elif k.endswith("ing"):
                if v[1] in X_ING:
                    X_ING[v[1]] += 1
                else:
                    X_ING[v[1]] = 1
            elif k.endswith("ed"):
                if v[1] in X_ED:
                    X_ED[v[1]] += 1
                else:
                    X_ED[v[1]] = 1
            elif k.endswith("er"):
                if v[1] in X_ER:
                    X_ER[v[1]] += 1
                else:
                    X_ER[v[1]] = 1
            elif k.endswith("est"):
                if v[1] in X_EST:
                    X_EST[v[1]] += 1
                else:
                    X_EST[v[1]] = 1
            elif k.endswith("tive"):
                if v[1] in X_TIVE:
                    X_TIVE[v[1]] += 1
                else:
                    X_TIVE[v[1]] = 1
            elif re.match("^[0-9]", k):
                if v[1] in X_NUM:
                    X_NUM[v[1]] += 1
                else:
                    X_NUM[v[1]] = 1
            elif k.endswith("s"):
                if v[1] in X_S:
                    X_S[v[1]] += 1
                else:
                    X_S[v[1]] = 1            
            else:
                if v[1] in hapax:
                    hapax[v[1]] += 1
                else:
                    hapax[v[1]] = 1

    # Compute smoothed probabilities & Take the log of each probability
    for i in cnt_emit:
        emission[i] = {}

        n = sum(cnt_emit[i].values())
        V = len(list(cnt_emit[i]))

        laplace_UNK = LAPLACE_EMIT*hapax.get(i, 1)
        laplace_ly = LAPLACE_EMIT*X_LY.get(i, 1)
        laplace_ing = LAPLACE_EMIT*X_ING.get(i, 1)
        laplace_ED = LAPLACE_EMIT*X_ED.get(i, 1)
        laplace_ER = LAPLACE_EMIT*X_ER.get(i, 1)
        laplace_EST = LAPLACE_EMIT*X_EST.get(i, 1)
        laplace_TIVE = LAPLACE_EMIT*X_TIVE.get(i, 1)
        laplace_NUM = LAPLACE_EMIT*X_NUM.get(i, 1)
        laplace_S = LAPLACE_EMIT*X_S.get(i, 1)
        laplace = laplace_UNK+laplace_ly+laplace_ing+laplace_ED+laplace_ER+laplace_EST+laplace_TIVE+laplace_NUM+laplace_S

        for (k, v) in cnt_emit[i].items():
            emission[i][k] = math.log(v+laplace) - math.log(n+laplace*(V+1))
        emission_UNK[i] = math.log(laplace) - math.log(n+laplace*(V+1))
        emission_LY[i] = math.log(laplace_ly) - math.log(n+laplace*(V+1))
        emission_ING[i] = math.log(laplace_ing) - math.log(n+laplace*(V+1))
        emission_ED[i] = math.log(laplace_ED) - math.log(n+laplace*(V+1))
        emission_ER[i] = math.log(laplace_ER) - math.log(n+laplace*(V+1))
        emission_EST[i] = math.log(laplace_EST) - math.log(n+laplace*(V+1))
        emission_TIVE[i] = math.log(laplace_TIVE) - math.log(n+laplace*(V+1))
        emission_NUM[i] = math.log(laplace_NUM) - math.log(n+laplace*(V+1))
        emission_S[i] = math.log(laplace_S) - math.log(n+laplace*(V+1))

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
                        if test[i][j] in emission[k2]:
                            p = v1[0] + v2 + emission[k2][test[i][j]]
                        elif test[i][j].endswith("ly"):
                            p = v1[0] + v2 + emission_LY[k2]
                        elif test[i][j].endswith("ing"):
                            p = v1[0] + v2 + emission_ING[k2]
                        elif test[i][j].endswith("ed"):
                            p = v1[0] + v2 + emission_ED[k2]
                        elif test[i][j].endswith("er"):
                            p = v1[0] + v2 + emission_ER[k2]
                        elif test[i][j].endswith("est"):
                            p = v1[0] + v2 + emission_EST[k2]
                        elif test[i][j].endswith("tive"):
                            p = v1[0] + v2 + emission_TIVE[k2]
                        elif re.match("^[0-9]", test[i][j]):
                            p = v1[0] + v2 + emission_NUM[k2]
                        elif test[i][j].endswith("s"):
                            p = v1[0] + v2 + emission_S[k2]
                        else:
                            p = v1[0] + v2 + emission_UNK[k2]

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