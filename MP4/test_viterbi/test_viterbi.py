# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 

See Piazza post @650 - This example is intended to show you that even though P("back" | RB) > P("back" | VB), 
the Viterbi algorithm correctly assigns the tag as VB in this context based on the entire sequence. 
"""
from utils import read_files, get_nested_dictionaries
import math

def main():
    test, emission, transition, output = read_files()
    emission, transition = get_nested_dictionaries(emission, transition)
    initial = transition["START"]
    prediction = []
    
    """WRITE YOUR VITERBI IMPLEMENTATION HERE"""
    trellis = []
    for i in test:
        for j in range(len(i)):
            trellis.append({})

            if j == 0:
                for (k, v) in initial.items():
                    trellis[j][k] = (math.log(v) + math.log(emission[k][i[j]]), "START")
            else:
                for (k1, v1) in trellis[j-1].items():
                    for (k2, v2) in transition[k1].items():
                        p = v1[0] + math.log(v2) + math.log(emission[k2][i[j]])
                        if k2 not in trellis[j] or p > trellis[j][k2][0]:
                            trellis[j][k2] = (p, k1)

        tag = "END"
        max = None
        for (k, v) in trellis[len(i)-1].items():
            if max == None or max < v[0]:
                tag = k
                max = v[0]

        for j in range(len(i)-1, -1, -1):
            prediction.insert(0, (i[j], tag))
            tag = trellis[j][tag][1]

    print('Your Output is:',prediction,'\n Expected Output is:',output)


if __name__=="__main__":
    main()