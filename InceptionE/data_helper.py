#coding:utf-8
import random
from collections import defaultdict
import numpy as np
from data_utils.kb_data_reader import read_kb_triples,read_kb_triples_reverse
import time


class DataHelper(object):
    def __init__(self,params,dataPath,vocabHelper):
        self.init_params(params,dataPath)
        self.vocabHelper=vocabHelper
        self.entity2id=vocabHelper.entity2id
        self.id2entity=vocabHelper.id2entity
        self.all_entities=list(range(len(self.id2entity)))
        self.num_entities=len(self.all_entities)
        self.relation2id=vocabHelper.relation2id
        self.id2relation=vocabHelper.id2relation
        self.process_datas()
        self.init_correct_entities()
    def init_correct_entities(self):
        # read triples
        test_triples, _, _ = read_kb_triples_reverse(self.test_dataPath)
        train_triples, _, _ = read_kb_triples_reverse(self.train_dataPath)
        valid_triples, _, _ = read_kb_triples_reverse(self.valid_dataPath)
        all_triples=train_triples + valid_triples + test_triples
        #query to correct tails
        correct_entities = defaultdict(set)
        for triple in all_triples:
            h, r, t = triple
            hid=self.entity2id.get(h)
            rid=self.relation2id.get(r)
            tid=self.entity2id.get(t)
            correct_entities[(hid,rid)].add(tid)
        self.correct_entities=correct_entities

    def init_params(self,params,dataPath):
        #datasets
        self.data_dir=params.data_dir
        self.dataPath=dataPath
        self.kb_path=params.kb_path
        self.train_dataPath=params.train_dataPath
        self.test_dataPath=params.test_dataPath
        self.valid_dataPath=params.valid_dataPath
        self.entityVocabPath=params.entityVocabPath
        self.relationVocabPath=params.relationVocabPath
        self.entityVectorPath=params.entityVectorPath
        self.relationVectorPath=params.relationVectorPath


    def read_datas(self):
        '''read datas'''
        triples,_,_=read_kb_triples_reverse(self.dataPath)
        return triples
    def process_datas(self):
        triples=self.read_datas()
        triples=self.vocabHelper.convert_triples_to_ids(triples)
        self.triples=np.array(triples)
        self.data_num=len(triples)
        print("data num:",self.data_num)

    def get_batch_datas(self,batch_triples):
        batch_triples=np.array(batch_triples)
        batch_datas={}
        batch_datas["head"]=batch_triples[:,0]
        batch_datas["relation"]=batch_triples[:,1]
        batch_datas["tail"]=batch_triples[:,2]
        return batch_datas

    def train_batch_generator(self,batch_size,shuffle=False):
        data_num=self.data_num
        ids=list(range(data_num))
        if shuffle:
            ids=random.sample(ids,data_num)
        batch_num=(data_num+batch_size-1)//batch_size
        for i in range(batch_num):
            bg=i*batch_size
            end=min((i+1)*batch_size,data_num)
            batch_ids=ids[bg:end]
            pos_batch_triples=self.triples[batch_ids]
            batch_datas={}
            batch_triples=list(pos_batch_triples)
            batch_datas["inputs"]=self.get_batch_datas(batch_triples)
            yield batch_datas



    def batch_generator(self, batch_size=1,start_index=0,test_num=None):
        if test_num is not None:
            triples=self.triples[start_index:start_index+test_num]
        else:
            triples=self.triples
        data_num = len(triples)
        batch_num = (data_num + batch_size - 1) // batch_size
        for i in range(batch_num):
            bg = i * batch_size
            end = min((i + 1) * batch_size, data_num)

            batch_triples = triples[bg:end]

            masks=[]
            for triple in batch_triples:
                h,r,t=triple
                corrects=list(self.correct_entities.get((h,r),[]))
                mask=np.zeros([self.num_entities])
                mask[corrects]=1
                mask[h]=1
                masks.append(mask)

            batch_datas = {}
            batch_datas["inputs"]=self.get_batch_datas(batch_triples)
            batch_datas["masks"]=masks
            yield batch_datas