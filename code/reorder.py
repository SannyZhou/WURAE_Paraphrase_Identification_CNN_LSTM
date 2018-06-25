# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: reorder.py

def reorder(thisNode, t, nextleaf, nextnode, opp, okids, opos):
    nnextleaf = nextleaf
    nnextnode = nextnode - 1
    nkids = okids

    pp = opp
    pos = opos

    kids = []
    row_max = t.kids.shape[0]
    for row in range(row_max):
        kids.append(t.kids[row][thisNode])
    kids = [value for index, value in enumerate(kids) if value]

    for k in range(2):
        kkk = t.isLeafnode[kids[k]]
        if kkk:
            pp[nnextleaf] = nextnode + 1
            nkids[nextnode][k] = nnextleaf + 1
            pos[nnextleaf] = t.pos[kids[k]]

            nnextleaf += 1
        else:
            pp[nnextnode] = nextnode + 1
            nkids[nextnode][k] = nnextnode + 1
            if kids[k] < len(t.pos):
                pos[nnextnode] = t.pos[kids[k]]
            pp, nnextleaf, nnextnode, nkids, pos = reorder(kids[k], t, nnextleaf, nnextnode, pp, nkids, pos)

    return pp, nnextleaf, nnextnode, nkids, pos