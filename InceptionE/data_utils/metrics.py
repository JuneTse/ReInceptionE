#coding:utf-8


def compute_hits_at_N(results,topN=10):
    '''
    计算hits@N
    :param top_results: list, [[(pred_score,label),……]
                               []]
    :param topN:
    :return:
    '''
    hits=[]
    for result in results:
        labels=[r[1] for r in result[:topN]]
        if sum(labels)>=1:
            hits.append(1)
        else:
            hits.append(0)
    right=sum(hits)
    num=len(hits)
    acc=right/num
    return num,right,acc

def compute_MR_MRR(results):
    mrs=[]
    mrrs=[]
    for result in results:
        for i,res in enumerate(result):
            pred,label=res
            if label==1:
                mrs.append(i+1)
                mrrs.append(1/(i+1))
    mr=sum(mrs)
    mrr=sum(mrrs)
    num=len(mrrs)
    return num,mr,mrr

def compute_MR_MRR_by_Rank(ranks):
    mrs=[]
    mrrs=[]
    for r in ranks:
        mrs.append(r+1)
        mrrs.append(1/(r+1))
    num = len(mrrs)
    mr=sum(mrs)/num
    mrr=sum(mrrs)/num

    return mr,mrr


def compute_Hits_K_by_Rank(ranks,topK=10):
    hits=[]
    for r in ranks:
        if r<topK:
            hits.append(1)
        else:
            hits.append(0)
    hits=sum(hits)/len(hits)
    return hits