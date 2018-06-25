# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: reformatTree.py

import numpy as np


def reformatTree(thisNode, t, upnext):
    # binarize
    kids =[]
    row_max = t.kids.shape[0]
    for row in range(row_max):
        kids.append(t.kids[row][thisNode])
    kids = [value for index, value in enumerate(kids) if value]
    kkk = t.isLeafnode[kids[0]]

    # only one child node and it is not a leaf node
    while len(kids) == 1 and kkk != 1:
        kkids = []
        row_max = t.kids.shape[0]
        for row in range(row_max):
            kkids.append(t.kids[row][kids[0]])
        kkids = [value for index, value in enumerate(kkids) if value]

        t.pp[kids[0]] = -1
        for id in kkids:
            t.pp[id] = thisNode + 1
        for row in range(len(kkids)):
            t.kids[row][thisNode] = kkids[row]

        kids = kkids
        kkk = t.isLeafnode[kids[0]]

    numnode = 0
    kkk = t.isLeafnode[kids[0]]
    if len(kids) == 1 and kkk:
        t.isLeafnode[thisNode] = 1
        t.pp[kids[0]] = -1
        row_max = t.kids.shape[0]
        for row in range(row_max):
            t.kids[row][thisNode] = 0
        inc = 0
        numnode = 1
    else:
        inc = 0
        for k in range(len(kids)):
            kkk = t.isLeafnode[kids[k]]
            # like postorder
            if not kkk:
                thisinc, thisnumnode, newt = reformatTree(kids[k], t, upnext + inc)
                inc += thisinc
                t = newt
                numnode += thisnumnode
            else:
                numnode += 1
        next = upnext + inc
        n = len(kids)
        last = kids[0]
        start = 1
        while n >= 2:
            if n == 2:
                next = thisNode
            else:
                next += 1
                inc += 1
            len_max = t.pp.shape
            tmp_pp = [0]
            while last >= len_max[0]:
                t.pp = np.concatenate((t.pp, tmp_pp))
                len_max = t.pp.shape
            t.pp[last] = next + 1
            t.pp[kids[start]] = next + 1
            row_max, column_max = t.kids.shape
            tmp_kids = np.zeros((row_max, 1), dtype=int)
            while next >= column_max:
                t.kids = np.concatenate((t.kids, tmp_kids), axis=1)
                row_max, column_max = t.kids.shape
            t.kids[0][next] = last
            t.kids[1][next] = kids[start]

            last = next
            start += 1
            n -= 1

    return inc, numnode, t

