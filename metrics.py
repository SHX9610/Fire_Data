import torch
def dice_score(pred,target):
    '''
    两张图片的交乘2除以他们的和
    :param pred: tensor with first dimension as batch
    :param target: tensor with first dimension as batch
    :return: score
    '''
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1*m2).sum()
    return (2.*intersection+smooth)/(m1.sum()+m2.sum()+smooth)
