# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: tree.py

class Tree:
    def __init__(self):
        self.pp = None
        self.nodeNames = None
        self.nodeFeatures = None
        self.nodeOut = None
        self.leafFeatures = None
        self.isLeafnode = None
        self.kids = None
        self.nodeLabels = None
        self.score = 0
        self.nodeScores = []
        self.pos = []

    def getTopNode(self):
        return self.pp.index(0)

    def getKids(self, node):
        return self.kids.index(node)

    def isLeaf(self, node):
        return self.isLeafnode[node]

    # def plotTree(self):


class Tree2:
    def __init__(self):
        self.pp = None
        self.nodeNames = None
        self.nodeFeatures = None
        self.nodeFeatures_unnormalized = None
        self.nodeFeatures_deep = None
        self.nodeFeatures_deep_u = None
        self.leafFeatures = None
        self.isLeafnode = None
        self.kids = None
        self.numkids = None
        self.nodeLabels = None
        self.score = 0
        self.nodeScores = []
        self.pos = []
        self.nodeZ = []
        self.nodeDelta_W1 = []
        self.nodeDelta_W2 = []
        self.nodeDelta_b1 = []
        self.nodeDelta_b2 = []
        self.parentDelta = []
        # self.catDelta = []
        # self.catDelta_out = []
        self.node_dl = []
        self.node_y1c1 = None
        self.node_y2c2 = None
        self.node_yc = []
        self.nums = []
        self.y1_unnormalized = None
        self.y2_unnormalized = None
        self.yc_sum = None

    def getTopNode(self):
        return self.pp.index(0)

    def getKids(self, node):
        return self.kids.index(node)

    def isLeaf(self, node):
        return self.isLeafnode[node]

    # def plotTree(self):


class Tree_torch:
    def __init__(self):
        self.nodeFeatures = None
        self.nodeFeatures_unnormalized = None
        self.kids = None
        self.counts = None

