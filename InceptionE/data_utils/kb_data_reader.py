#coding:utf-8
import numpy as np
from collections import defaultdict


def read_word2id(path):
    '''read entity id
    '''
    with open(path,encoding="utf-8") as fin:
        word2id={}
        id2word={}
        for line in fin:
            line=line.strip().split()
            assert len(line)==2
            e,id=line
            id=int(id)
            word2id[e]=id
            id2word[id]=e
        return word2id,id2word

def read_word2vec(path):
    vectors=[]
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip()
            vec=[float(v) for v in line.split()]
            size=len(vec)
            # assert size==100
            vectors.append(vec)
    vectors=np.array(vectors)
    print("%s vector size:%s"%(path,vectors.shape))
    return vectors

def read_kb_triples(kb_path):
    '''read triples'''
    t2heads=defaultdict(dict)
    h2tails=defaultdict(dict)
    triples=[]
    with open(kb_path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip().split()
            assert len(line)==3,line
            h,r,t=line
            triples.append([h,r,t])
            t2heads[t][h]=r
            h2tails[h][t]=r
    return triples,h2tails,t2heads


def read_kb_triples_reverse(kb_path):
    '''read triples'''
    t2heads=defaultdict(dict)
    h2tails=defaultdict(dict)
    triples=[]
    with open(kb_path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip().split()
            assert len(line)==3,line
            h,r,t=line
            triples.append([h,r,t])
            t2heads[t][h]=r
            h2tails[h][t]=r
            #reverse
            r2="%s_reverse"%r
            triples.append([t,r2,h])
            t2heads[h][t]=r2
            h2tails[t][h]=r2

    return triples,h2tails,t2heads



