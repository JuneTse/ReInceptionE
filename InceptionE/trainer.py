#coding:utf-8
import numpy as np
import time
from data_evaluate import compute_metrics
import os
import logging
import tensorflow as tf

t=time.localtime()
mon=t[1]
date=t[2]
h=t[3]
m=t[4]
def getLogger(path,data_dir,name="Logger",mode="a"):
    logger=logging.Logger(name)
    logger.setLevel(logging.INFO)
    name="%s-%s-%s-%s"%(mon,date,name,data_dir)
    filename=os.path.join(path,name)
    fh=logging.FileHandler(filename=filename,mode=mode)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info("add logger")
    return logger

class Trainer(object):
    def __init__(self, model, params):
        self.model = model
        self.init_params(params)
        data_dir=params.data_dir
        name = model.__class__.__name__
        # save path
        self.ckpt_path = os.path.join(params.weight_path, "%s_%s" % (name, data_dir))
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        self.log_path = os.path.join(params.log_path, model.__class__.__name__)
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.log = getLogger(path=self.log_path, data_dir=data_dir,name=model.__class__.__name__)
    def init_params(self, params):
        self.batch_size = params.batch_size
        self.global_step = 0
        self.keep_prob = params.keep_prob
        # lr
        self.init_lr = params.lr
        self.lr_decay = params.lr_decay
        self.lr_decay_step = params.lr_decay_step
        self.warm_up_epoch=params.warm_up_step

        self.eval_step_num = params.eval_step_num or 100
        self.eval_epoch_num = params.eval_epoch_num or 1
        self.log_step_num = params.log_step_num or 100
        self.saver = None
    def get_feed_dict(self,batch_datas,mode="train"):
        feed_dict={}
        feed_dict[self.model.features["inputs"]["head"]] = batch_datas["inputs"]["head"]
        feed_dict[self.model.features["inputs"]["tail"]] = batch_datas["inputs"]["tail"]
        feed_dict[self.model.features["inputs"]["relation"]] = batch_datas["inputs"]["relation"]
        return feed_dict
    def train(self, sess, data_helper, eval_data_helper, test_data_helper=None, iter_num=50, shuffle=True):
        for epoch in range(iter_num):
            self.log.info("epoch: %s" % epoch)
            data_gen = data_helper.train_batch_generator(batch_size=self.batch_size, shuffle=shuffle)
            total_loss = 0
            if epoch>=self.warm_up_epoch:
                exp=(epoch // self.lr_decay_step)
                self.lr = self.init_lr * (np.power(self.lr_decay, exp))
                if self.lr<5e-5:
                    exp = (epoch // self.lr_decay_step)%10
                    self.lr=1e-4*(np.power(self.lr_decay, exp))
            else:
                self.lr=self.init_lr*((self.global_step+1)/(data_helper.data_num*self.warm_up_epoch/self.batch_size+1))

            epoch_start_time=time.time()
            for batch_datas in data_gen:
                feed_dict = self.get_feed_dict(batch_datas)
                feed_dict[self.model.lr] = self.lr
                if "keep_prob" in self.model.features:
                    feed_dict[self.model.features["keep_prob"]] = self.keep_prob

                train_outputs =  sess.run(self.model.train_outputs, feed_dict=feed_dict)
                loss = train_outputs["loss"]
                total_loss += loss
                self.global_step += 1
            if (epoch+1) % self.eval_epoch_num == 0:
                if (epoch+1) %5==0:
                    self.evaluate_model(sess, eval_data_helper, test_num=None)
                    if test_data_helper is not None:
                        self.evaluate_model(sess, test_data_helper, test_num=None)
                self.save_weights(sess, global_step=epoch)
            epoch_end_time = time.time()
            print("epoch: %s, lr: %s ,total loss :%s, time: %s"%(epoch,self.lr,total_loss, epoch_end_time-epoch_start_time))
    def evaluate_model(self,sess, test_dataHelper, batch_size=64,test_num=100):
        ranks = []
        # predict
        gen = test_dataHelper.batch_generator(batch_size=batch_size)
        for batch_datas in gen:
            if batch_datas is None:
                continue
            batch_pred_scores = self.predict_batch(sess,batch_datas)
            masks = batch_datas["masks"]
            inputs = batch_datas["inputs"]
            batch_tails = inputs["tail"]
            for tail, mask, preds in zip(batch_tails, masks, batch_pred_scores):
                preds=-preds
                tail_score = preds[tail]
                preds = preds + mask * (10000)
                preds[tail] = tail_score
                # rank
                results = np.argsort(preds, axis=0)
                rank = np.where(results == tail)[0][0]
                ranks.append(rank)

        print("number of test data:",len(ranks))
        mr,mrr,hit_1,hit_3,hit_10,hit_30=compute_metrics(ranks,is_print=True)
        self.log.info("Result: %s\t%s\t%s\t%s\t%s"%(mr,mrr,hit_1,hit_3,hit_10))
        return mrr,hit_10

    def predict_batch(self, sess, batch_datas):
        feed_dict = self.get_feed_dict(batch_datas, mode="predict")
        if "keep_prob" in self.model.features:
            feed_dict[self.model.features["keep_prob"]] = 1.0
        output = sess.run(self.model.scores_predict, feed_dict)
        output = output[:, :]
        return output

    def save_weights(self, sess, global_step=None, saver=None):
        if saver is None:
            if self.saver is None:
                self.saver = tf.train.Saver(max_to_keep=100)
            saver = self.saver
        saver.save(sess, save_path=os.path.join(self.ckpt_path, "weights.ckpt"), global_step=global_step)

    def restore_last_session(self, sess):
        '''load model'''
        saver = tf.train.Saver()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore params from %s" % ckpt.model_checkpoint_path)
        else:
            print("fail to restore..., ckpt:%s" % ckpt)

