# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: runReformatTree.py

import tree
import numpy as np
from reformatTree import *
from reorder import *


def runReformatTree(allSTree, allSNum, allSStr, allSPOS):
    numinstance = len(allSTree)
    allSKids = [None] * numinstance
    empty = []
    for instance in range(numinstance):
        n = len(allSTree[instance])

        # the number of words in sentence
        cnt = 0
        for j in range(len(allSStr[instance])):
            if allSStr[instance][j] != []:
                cnt += 1

        # the number of words in sentence < 2
        if cnt < 2:
            if cnt == 1:
                allSNum[instance] = allSNum[instance][-1]
                allSStr[instance] = allSStr[instance][-1]
                allSTree[instance] = allSTree[instance][-1]
                allSPOS[instance] = allSPOS[instance][-1]
            else:
                empty.append(instance)
            continue

        t = tree.Tree()
        t.pp = np.zeros(n, dtype=int)
        t.pp[0:n] = allSTree[instance]
        mostkids = np.max(np.bincount(allSTree[instance]))
        t.kids = np.zeros([mostkids, n], dtype=int)

        for i in range(0, n):
            tempkids = [tmp_index for tmp_index, value in enumerate(allSTree[instance]) if value == i + 1]
            for row in range(len(tempkids)):
                t.kids[row][i] = tempkids[row]

        t.leafFeatures = np.zeros(n)
        leafs = [tmp_index for tmp_index, value in enumerate(allSNum[instance]) if value > 0]
        t.isLeafnode  = np.zeros(2 * n, dtype=int)
        for index in range(2*n):
            if index in leafs:
                t.isLeafnode[index] = 1

        t.pos = allSPOS[instance]

        for i in range(len(leafs)):
            t.leafFeatures[leafs[i]] = allSNum[instance][leafs[i]]

        inc, numnode, newt = reformatTree(0, t, n)
        # print(inc, numnode, newt)
        opp  = np.zeros(2 * numnode - 1,dtype=int)
        okids = np.zeros([2 * numnode - 1, 2], dtype=int)
        opos  = [[None] for i in  range(2 * numnode - 1)]

        pp, nnextleaf, nnextnode, nkids, pos = reorder(0, newt, 0, 2 * numnode - 2, opp, okids, opos)
        newnum = np.zeros(numnode, dtype=int)
        newstr = [None] * numnode
        next = 0
        for i in range(len(allSNum[instance])):
            if allSNum[instance][i] > 0:
                newnum[next] = allSNum[instance][i]
                newstr[next] = allSStr[instance][i]
                next += 1

        allSNum[instance] = newnum
        allSStr[instance] = newstr
        allSTree[instance] = pp
        allSKids[instance] = nkids
        allSPOS[instance] = pos

    return allSNum, allSStr, allSTree, allSKids, allSPOS
