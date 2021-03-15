#coding:utf-8
from collections import defaultdict
import numpy as np
from data_utils.kb_data_reader import read_word2id,read_word2vec


class VocabHelper(object):
    '''Read Vocab datas'''
    def __init__(self,params):
        self.init_params(params)
        self.trainPath=params.train_dataPath
        self.read_datas()

    def init_params(self,params):
        #vocab path
        self.entityVocabPath=params.entityVocabPath
        self.relationVocabPath=params.relationVocabPath

        self.entity_vector_path=params.entity_vector_path
        self.relation_vector_path=params.relation_vector_path


    def read_datas(self):
        self.entity2id,self.id2entity=read_word2id(self.entityVocabPath)
        self.relation2id,self.id2relation=read_word2id(self.relationVocabPath)
        self.entity_embeddings = np.array(read_word2vec(self.entity_vector_path)).astype("float32")
        self.relation_embeddings=np.array(read_word2vec(self.relation_vector_path)).astype("float32")

        print("entity embedding shape:",self.entity_embeddings.shape)
        print("relation embedding shape:",self.relation_embeddings.shape)
    def convert_entities_to_ids(self,entities):
        ids=[]
        for entity in entities:
            i=self.entity2id.get(entity)
            ids.append(i)
        return ids
    def convert_relations_to_ids(self,relations):
        ids=[]
        for relation in relations:
            i=self.relation2id.get(relation)
            ids.append(i)
        return ids

    def convert_triples_to_ids(self,triples):
        triple_ids=[]
        for h,r,t in triples:
            hid=self.entity2id.get(h)
            tid=self.entity2id.get(t)
            rid=self.relation2id.get(r)
            triple_ids.append([hid,rid,tid])
        return triple_ids
    def convert_ids_to_triple(self,triple_ids):
        hid,rid,tid=triple_ids
        h=self.id2entity[hid]
        r=self.id2relation[rid]
        t=self.id2entity[tid]
        return (h,r,t)


class VocabHelperReverse(VocabHelper):
    def read_datas(self):
        self.entity2id,self.id2entity=read_word2id(self.entityVocabPath)
        relation2id,id2relation=read_word2id(self.relationVocabPath)
        self.relation2id={}
        self.id2relation={}
        self.relation2id.update(relation2id)
        self.id2relation.update(id2relation)
        self.relation2reverse={}

        for rid1,r in id2relation.items():
            rid=len(self.id2relation)
            self.relation2reverse[rid1]=rid
            self.relation2reverse[rid]=rid1
            r="%s_reverse"%r
            self.relation2id[r]=rid
            self.id2relation[rid]=r

        self.entity_embeddings = np.array(read_word2vec(self.entity_vector_path)).astype("float32")
        relation_embeddings=np.array(read_word2vec(self.relation_vector_path)).astype("float32")
        self.relation_embeddings=np.concatenate([relation_embeddings,relation_embeddings],axis=0)