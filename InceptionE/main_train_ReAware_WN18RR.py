#coding:utf-8
import tensorflow as tf
from data_helper_vocab import VocabHelperReverse
from data_helper import DataHelper
from models_InceptionE import InceptionE
from trainer import Trainer
from data_evaluate import evaluate_model
from data_utils.params import Params


data_dir="WN18RR"
kb_params = {
    "share_emb":False,
    #data
    "data_dir":data_dir,
    "entity_vocab_size": 40943,
    "relation_vocab_size": 11,
    "triple_num":86835,
    #model
    "emb_dim":100,
    #loss
    "l2_reg_lambda":1e-5,
    "gamma":5,
    #train
    "optimizer":tf.train.AdamOptimizer,
    "keep_prob":0.6,
    "batch_size":256,
    "lr": 0.0002,
    "lr_decay": 0.95,
    "lr_decay_step": 1,
    "warm_up_step": 5
    }
params = Params(data_dir)

if __name__=="__main__":
    params=Params(data_dir)
    params.update(kb_params)

    vocabHelper=VocabHelperReverse(params=params)
    #datasets
    valid_dataHelper=DataHelper(dataPath=params.valid_dataPath, params=params, vocabHelper=vocabHelper)
    test_dataHelper = DataHelper(dataPath=params.test_dataPath, params=params, vocabHelper=vocabHelper)
    train_dataHelper=DataHelper(dataPath=params.train_dataPath, params=params, vocabHelper=vocabHelper)
    entities = vocabHelper.entity2id.keys()

    entity_embeddings=vocabHelper.entity_embeddings
    relation_embeddings=vocabHelper.relation_embeddings

    with tf.Session() as sess:
        #model
        model=InceptionE(params,
                        entity_embedding=vocabHelper.entity_embeddings,
                        relation_embedding=vocabHelper.relation_embeddings)
        #train
        trainer=Trainer(model, params)
        sess.run(tf.global_variables_initializer())
        # trainer.restore_last_session(sess)
        trainer.train(sess,data_helper=train_dataHelper,eval_data_helper=valid_dataHelper,test_data_helper=test_dataHelper,iter_num=100)
        #predict
        evaluate_model(sess,trainer,test_dataHelper)