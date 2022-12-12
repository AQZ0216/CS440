"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    dict_word = {}
    dict_tag = {}

    for i in train:
        for j in i:
            if j[1] in dict_tag:
                dict_tag[j[1]] += 1
            else:
                dict_tag[j[1]] = 1

            if j[0] not in dict_word:
                dict_word[j[0]] = {}

            if j[1] in dict_word[j[0]]:
                dict_word[j[0]][j[1]] += 1
            else:
                dict_word[j[0]][j[1]] = 1

    most_seen_tag = "START"
    max = 0
    for (k, v) in dict_tag.items():
        if max < v:
            most_seen_tag = k
            max = v

    output = []
    for i in range(len(test)):
        output.append([])
        for j in test[i]:
            if j not in dict_word:
                output[i].append((j, most_seen_tag))
            else:
                tag = "START"
                max = 0

                for (k, v) in dict_word[j].items():
                    if max < v:
                        tag = k
                        max = v
                output[i].append((j, tag))
    
    return output