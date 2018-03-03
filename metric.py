"""
  input should be a torch.Tensor
  it would be easy to implement in numpy, I think.
"""

import torch

def pix_acc(pred_labels, gt_labels):
    """
        return the accuracy of all pixels
        pred_labels and gt_labels should be batch x w x h
    """
    pix = pred_labels.size()[1]*pred_labels.shape[2]
    return torch.sum(((pred_labels-gt_labels) == 0).view(2,4).sum(1).type(torch.FloatTensor)/pix)

def mean_pix_acc(pred_labels, gt_labels, class_num):
    """
        
    """

    mean_cls_acc = 0

    for cls in range(class_num):
        # extract the indices of the labels(number) of cls
        mask = gt_labels == cls
        mean_cls_acc += (pred_labels[mask] == cls).sum()/mask.sum()

    return mean_cls_acc/class_num

def seg_metric(pred_labels, gt_labels, class_num):
    """
        adding 1 to set all class incluging background to
        make correct class subtract to 0, and count the non-zero value
    
        input should be batch_size * height * width
    """    
    pix = pred_labels.shape[1]*pred_labels.shape[2]
    all_pix_acc = ((pred_labels-gt_labels) == 0).sum()/pix

    mean_cls_acc = 0

    for cls in range(class_num):
        # extract the indices of the labels(number) of cls
        mask = gt_labels == cls

        """
        # count cls-th class labels
        cls_pix_num = mask.sum()
        
        # count predicted class that belongs to cls-th class, which mean correct.
        pred_correct_cls = (pred_labels[mask] == cls).sum()
        
        class_predict_acc = pred_correct_cls/cls_pix_num
        """
        mean_cls_acc += (pred_labels[mask] == cls).sum()/mask.sum()
    
    return {"pix acc": all_pix_acc, "mean pix acc" : mean_cls_acc/class_num, "mean IoU":0}

def evaluate(label_trues, label_preds, n_class, eval_list=["IoU"]):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu
