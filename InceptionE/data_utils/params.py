#coding:utf-8

class ParamsDict(dict):
    def __init__(self, d=None):
        if d is not None:
            for k,v in d.items():
                self[k] = v
        return super().__init__()
    def __key(self, key):
        return "" if key is None else key.lower()
    def __str__(self):
        import json
        return json.dumps(self)
    def __setattr__(self, key, value):
        self[self.__key(key)] = value
    def __getattr__(self, key):
        return self.get(self.__key(key))
    def __getitem__(self, key):
        return super().get(self.__key(key))
    def __setitem__(self, key, value):
        return super().__setitem__(self.__key(key), value)

class Params(ParamsDict):
    def __init__(self,data_dir):
        # data paths
        self.kb_path="datasets/%s/train.txt" % data_dir
        self.train_dataPath="datasets/%s/train.txt" % data_dir
        self.valid_dataPath="datasets/%s/valid.txt" % data_dir
        self.test_dataPath="datasets/%s/test.txt" % data_dir
        # vocab
        self.entityVocabPath="datasets/%s/entity2id.txt" % data_dir
        self.relationVocabPath="datasets/%s/relation2id.txt" % data_dir
        # embedding
        self.entity_vector_path="datasets/%s/entity2vec.txt" % data_dir
        self.relation_vector_path="datasets/%s/relation2vec.txt" % data_dir
        # embedding
        self.emb_dim=200
        #model
        self.hidden_dim=100
        #loss
        self.l2_reg_lambda=0.001
        # train
        self.keep_prob=0.8
        self.batch_size=128
        self.lr=0.001
        self.lr_decay=0.995
        self.lr_decay_step=5
        self.warm_up_step=5
        # log
        self.eval_step_num=500
        self.eval_epoch_num=1
        self.weight_path="./weights/"
        self.log_path="./weights/"




