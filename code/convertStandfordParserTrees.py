# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: convertStandfordParserTrees.py

from getWordEmbedding import *
import re
from runReformatTree import *
import pickle
from fileprocess import *


def convertStanfordParserTrees(parsedfile):
    global wordEmbedding
    # wordEmbedding = get_word_embedding()

    allSNum = []
    allSStr = []
    allSOStr = []
    allSPOS = []
    allSTree = []

    parsedFid = open(parsedfile, 'r')
    parsedLines = parsedFid.readlines()
    parsedFid.close()
    # print(parsedLines[:25])

    sNum = []
    # n-th of sentence that can't be parsed
    c = []
    # n-th of sentence with no pcfg fallback
    cc = []

    for i in range(0, len(parsedLines) - 1):
        parsedLines[i] = parsedLines[i].lstrip(' ').rstrip('\n')

        # next sentence
        if parsedLines[i] == '':
            # print('next sentence')
            continue

        # skip the sentence which can not be parsed
        if parsedLines[i].lower() == 'SENTENCE_SKIPPED_OR_UNPARSABLE'.lower() or parsedLines[i][1] == '<':
            allSNum.append([])
            allSStr.append([])
            allSOStr.append([])
            allSPOS.append([])
            allSTree.append([])
            c.append(len(allSNum) - 1)
            continue

        # skip the sentence with no fallback
        if parsedLines[i].lower() == 'Sentence skipped: no PCFG fallback.'.lower():
            cc.append(len(allSNum) - 1)
            continue

        line = parsedLines[i].split(' ')
        if line == ['']:
            continue

        if sNum == []:
            sNum = [-1]
            sStr = [[]]
            sOStr = ['']

            # get the pos
            pattern = re.compile(r'([A-Z]+)')
            posTag = pattern.findall(parsedLines[i])
            sPOS = [posTag[0]]

            if parsedLines[i][0:3] == '((':
                tmp = line
                line = ['('].append(tmp[0][1:])
                line = line.extend(tmp[1:])

            # the tree\parent of  current node
            sTree = [0]
            lastParents = [1]
            currentParent = 1
            if len(line) > 2:
                line = line[2:]
            else:
                continue

        lineLength = len(line)
        s = 0
        if isinstance(line, str):
            line = [line]

        while s < lineLength:
            nextIsWord = s < lineLength-1 and ((line[s+1][-1] == ')') or (not(line[s+1][0] == '(') and s < lineLength-2))
            startsBranch = (line[s][0] == '(')
            if startsBranch and not nextIsWord:
                sTree.append(currentParent)
                sStr.append([])
                sOStr.append('')
                sPOS.append(line[s][1:])
                sNum.append(-100)
                currentParent = len(sNum)
                lastParents.append(currentParent)
                s += 1
                continue

            if startsBranch and nextIsWord:
                numWords = 1
                left_b = [loc.start() for loc in re.finditer('\(', line[s + numWords])]
                right_b = [loc.start() for loc in re.finditer('\)', line[s + numWords])]

                while len(right_b) <= len(left_b):
                    word = line[s + numWords]
                    sStr.append(word.lower())
                    sOStr.append(word)
                    sTree.append(currentParent)
                    sPOS.append(line[s][1:])
                    sNum.append(1)

                    numWords += 1
                    assert s + numWords <= lineLength
                    left_b = [loc.start() for loc in re.finditer('\(', line[s + numWords])]
                    right_b = [loc.start() for loc in re.finditer('\)', line[s + numWords])]

                if left_b != []:
                    word = line[s + numWords][left_b[0] + 1:right_b[0]]
                else:
                    word = line[s + numWords][:right_b[0]]

                sStr.append(word.lower())
                sOStr.append(word)
                sTree.append(currentParent)
                sPOS.append(line[s][1:])
                sNum.append(1)
                s += numWords + 1
                if len(lastParents) < len(right_b) - len(left_b):
                    lastParents = []
                elif len(lastParents) == len(right_b) -len(left_b):
                    lastParents = lastParents[0]
                else:
                    lastParents = lastParents[0 : len(lastParents) + len(left_b) - len(right_b) + 1]

                if lastParents == []:
                    assert len(sNum) == len(sStr)
                    assert len(sNum) == len(sPOS)
                    assert len(sNum) == len(sTree)

                    allSNum.append(sNum)
                    allSStr.append(sStr)
                    allSOStr.append(sOStr)
                    allSPOS.append(sPOS)
                    allSTree.append(sTree)

                    s += 1
                    sNum = []
                    sStr = []
                    sOStr = []
                    sPOS = []
                    sTree = []

                    continue
                currentParent = lastParents[-1]
                continue

    allSNum, allSStr, allSTree, allSKids, allSPOS = runReformatTree(allSTree, allSNum, allSStr, allSPOS)
    data = {'allSNum': allSNum,
            'allSStr': allSStr,
            'allSTree': allSTree,
            'allSKids': allSKids,
            'allSPOS': allSPOS}
    return data


if __name__ == '__main__':
    #parsedfile = '../data/sentence_msr_paraphrase_trainparsed.txt'
    parsedfile = 'tmp_eng_tree.list'
    data = convertStanfordParserTrees(parsedfile)
    print(data)
    # data = {'allSNum': allSNum,
    #         'allSStr': allSStr,
    #         'allSTree': allSTree,
    #         'allSKids': allSKids,
    #         'allSPOS': allSPOS}
    #output_file = open('../data/convertStanfordParserTrees.pkl', 'wb')
    #pickle.dump(data, output_file)
    #output_file.close()

