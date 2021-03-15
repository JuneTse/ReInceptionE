#coding:utf-8
import time
import numpy as np
from data_utils.metrics import compute_Hits_K_by_Rank,compute_MR_MRR_by_Rank

def compute_metrics(ranks,is_print=True):
    '''compute Hits@N/MRR'''
    mr,mrr=compute_MR_MRR_by_Rank(ranks)
    hit_1=compute_Hits_K_by_Rank(ranks,topK=1)
    hit_3=compute_Hits_K_by_Rank(ranks,topK=3)
    hit_10=compute_Hits_K_by_Rank(ranks,topK=10)
    hit_30=compute_Hits_K_by_Rank(ranks,topK=50)
    if is_print:
        print("MR   MRR hit@1    hit@3   hit@10")
        print("%s\t%s\t%s\t%s\t%s"%(mr,mrr,hit_1,hit_3,hit_10))
    return mr,mrr,hit_1,hit_3,hit_10,hit_30

def rank_args(preds,labels):
    #rank
    results=np.argsort(preds,axis=-1)
    ranks=np.where(results==labels)[1]
    args=[res[:100] for res in results]
    return ranks,args

def evaluate_model(sess,trainer,test_dataHelper,batch_size=2):
    ranks=[]
    #predict
    gen=test_dataHelper.batch_generator(batch_size=batch_size,start_index=0,test_num=None)
    i=0
    for batch_datas in gen:
        if batch_datas is None:
            continue
        i+=1
        batch_pred_scores=trainer.predict_batch(sess,batch_datas)
        masks=batch_datas["masks"]
        inputs=batch_datas["inputs"]
        batch_heads=inputs["head"]
        batch_tails = inputs["tail"]
        batch_ranks=[]
        for head,tail,mask,preds in zip(batch_heads,batch_tails,masks,batch_pred_scores):
            preds=-preds
            tail_score=preds[tail]
            preds=preds+mask*(1000000)
            preds[tail]=tail_score
            # rank
            results = np.argsort(preds, axis=0)
            rank= np.where(results == tail)[0][0]
            batch_ranks.append(rank)

        ranks.extend(batch_ranks)

    tail_ranks=[ranks[i] for i in range(0,len(ranks),2)]
    head_ranks=[ranks[i] for i in range(1,len(ranks),2)]

    #compute metrics
    print("head ranking ....")
    compute_metrics(head_ranks)
    print("tail ranking ....")
    compute_metrics(tail_ranks)
    print("Total ranking ...")
    compute_metrics(ranks)



