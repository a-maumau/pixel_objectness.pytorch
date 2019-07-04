"""
  input should be a torch.Tensor
  it would be easy to implement in numpy, I think.
"""
import torch
import torch.nn as nn
import numpy as np

eps = 1e-8
CPU = torch.device('cpu')

def pixel_accuracy(pred_labels, gt_labels, size_average=True, map_device=CPU):
    """Return the pixel accuracy        
        args
            pred_labels and gt_labels should be batch x w x h

        return
            mean pixel accuracy.
            if the size_mean is False, it returns a vector of pixel accuracy
    """
    batch_size = pred_labels.shape[0]
    w = pred_labels.shape[1]
    h = pred_labels.shape[2]
    
    # same thing.
    if size_average:
        return torch.mean(torch.mean((pred_labels.to(device=map_device, dtype=torch.long)==gt_labels).view(batch_size, -1).to(device=map_device, dtype=torch.float32), dim=1), dim=0).item()
    else:
        result = []
        batch_result = torch.mean((pred_labels.to(device=map_device, dtype=torch.long)==gt_labels).view(batch_size, -1).to(device=map_device, dtype=torch.float32), dim=1)
        for b in range(batch_size):
            result.append(batch_result[b].item())

        return result

def precision(pred_labels, gt_labels, class_num=2, size_average=True, only_class=None, ignore=[255], exclude_non_appear_class=True, map_device=CPU):
    """
    precision: TP/(TP+FP)

        ignore: list or int
            it will be ignored in the evaluation.
            exception is if only_class is not None.
            it will evaluate the only_class class.

        exclude_non_appear_class=True: bool
            if this is true, it will exclude the non-appearing class in the gt_labels.
            setting this to False case is if you want to return the non-prediction is correct and score is 1.0.
            when size_average is False, it return the list of metric score without the exluded class.
    """

    result = {}
    batch_size = pred_labels.shape[0]
    if isinstance(ignore, int):
        ignore = [ignore]

    if only_class:
        assert isinstance(only_class, int), "only_class should int"

        count = 0
        batch_result = []

        class_id = only_class

        class_mask = (gt_labels==class_id).view(batch_size, -1).to(device=map_device, dtype=torch.long) # 0,1 mask
        pred_class = (pred_labels==class_id).view(batch_size, -1).to(device=map_device, dtype=torch.long) # 0,1 mask
        cls_gt = torch.sum(class_mask, dim=1)

        TP = torch.sum((pred_class*class_mask).view(batch_size, -1), dim=1)
        TPFP = torch.sum(pred_class.view(batch_size, -1), dim=1)

        # to avoid error at all-zero mask
        for batch_index in range(batch_size):
            if cls_gt[batch_index] == 0 and exclude_non_appear_class:
                continue
            elif TPFP[batch_index] == 0 and cls_gt[batch_index] == 0: 
                batch_result.append(1.0)
                count += 1
            elif TPFP[batch_index] == 0 and cls_gt[batch_index] != 0:
                batch_result.append(0.0)
                count += 1
            else:
                batch_result.append(float(TP[batch_index])/float(TPFP[batch_index]))
                count += 1

        if size_average:
            result["class_{}".format(class_id)] = sum(batch_result)/count
        else:
            result["class_{}".format(class_id)] = batch_result

    else:
        for class_id in range(class_num):
            if class_id in ignore:
                continue

            count = 0
            batch_result = []

            class_mask = (gt_labels==class_id).view(batch_size, -1).to(device=map_device, dtype=torch.long) # 0,1 mask
            pred_class = (pred_labels==class_id).view(batch_size, -1).to(device=map_device, dtype=torch.long) # 0,1 mask
            cls_gt = torch.sum(class_mask, dim=1)

            TP = torch.sum((pred_class*class_mask).view(batch_size, -1), dim=1).cpu()
            TPFP = torch.sum(pred_class.view(batch_size, -1), dim=1).cpu()

            # to avoid error at all-zero mask
            for batch_index in range(batch_size):
                if cls_gt[batch_index] == 0 and exclude_non_appear_class:
                    continue
                elif TPFP[batch_index] == 0 and cls_gt[batch_index] == 0: 
                    batch_result.append(1.0)
                    count += 1
                elif TPFP[batch_index] == 0 and cls_gt[batch_index] != 0:
                    batch_result.append(0.0)
                    count += 1
                else:
                    batch_result.append(float(TP[batch_index])/float(TPFP[batch_index]))
                    count += 1

            if size_average:
                result["class_{}".format(class_id)] = sum(batch_result)/count
            else:
                result["class_{}".format(class_id)] = batch_result

    return result

def jaccard_index(pred_labels, gt_labels, class_num=2, size_average=True, only_class=None, ignore=[255], exclude_non_appear_class=True, map_device=CPU):
    """
        pred_labels and gt_labels should be batch x w x h

        known as IoU
    """
    result = {}
    batch_size = pred_labels.shape[0]
    if isinstance(ignore, int):
        ignore = [ignore]
    
    if only_class is not None:
        assert isinstance(only_class, int), "only_class should int"

        count = 0
        batch_result = []

        class_id = only_class

        class_mask = (gt_labels==class_id).view(batch_size, -1).to(device=map_device, dtype=torch.long) # 0,1 mask
        pred_class = (pred_labels==class_id).view(batch_size, -1).to(device=map_device, dtype=torch.long) # 0,1 mask
        
        intersection = torch.sum(pred_class*class_mask, dim=1)
        u_pred = torch.sum(pred_class, dim=1)
        u_gt = torch.sum(class_mask, dim=1)

        # to avoid error at all-zero mask
        for batch_index in range(batch_size):
            denominator = u_pred[batch_index] + u_gt[batch_index] - intersection[batch_index]
            if u_gt[batch_index] == 0 and exclude_non_appear_class:
                    continue
            elif denominator == 0:
                batch_result.append(1.0)
                count += 1
            else:
                batch_result.append(float(intersection[batch_index].cpu().data)/float(denominator.cpu().data))
                count += 1

        if size_average:
            result["class_{}".format(class_id)] = sum(batch_result)/count
        else:
            result["class_{}".format(class_id)] = batch_result

    else:
        for class_id in range(class_num):
            if class_id in ignore:
                continue

            count = 0
            batch_result = []

            class_mask = (gt_labels==class_id).view(batch_size, -1).to(device=map_device, dtype=torch.long) # 0,1 mask
            pred_class = (pred_labels==class_id).view(batch_size, -1).to(device=map_device, dtype=torch.long) # 0,1 mask
            
            intersection = torch.sum(pred_class*class_mask, dim=1)
            u_pred = torch.sum(pred_class, dim=1)
            u_gt = torch.sum(class_mask, dim=1)

            # to avoid error at all-zero mask
            for batch_index in range(batch_size):
                denominator = u_pred[batch_index] + u_gt[batch_index] - intersection[batch_index]
                if u_gt[batch_index] == 0 and exclude_non_appear_class:
                    continue
                elif denominator == 0:
                    batch_result.append(1.0)
                    count += 1
                else:
                    batch_result.append(float(intersection[batch_index].cpu().data)/float(denominator.cpu().data))
                    count += 1

            if size_average:
                result["class_{}".format(class_id)] = sum(batch_result)/count
            else:
                result["class_{}".format(class_id)] = batch_result

    return result

def dice_score(pred_labels, gt_labels):
    """
        binary class only
        return the accuracy of all pixels
        pred_labels and gt_labels should be batch x w x h
    """
    result = []

    w = pred_labels.size()[1]
    h = pred_labels.size()[2]
    batch_size = pred_labels.size()[0]

    TP = torch.sum((pred_labels.type(torch.long)*gt_labels).view(batch_size, -1), dim=1).type(torch.FloatTensor)
    FPFN = torch.sum((pred_labels.type(torch.long)!=gt_labels).view(batch_size, -1), dim=1).type(torch.FloatTensor)

    # to avoid error at all-zero mask
    for batch_index in range(batch_size):
        denominator = TP[batch_index]*2+FPFN[batch_index]
        if denominator == 0:
            result.append(1.0)
        else:
            result.append((TP[batch_index]*2)/denominator)

    return torch.FloatTensor(result)

# from http://forums.fast.ai/t/understanding-the-dice-coefficient/5838
class SoftDiceLoss(nn.Module):
    """
        pred is size of batch x 2 channel x w x h
    """
    def __init__(self, weight=None, size_average=True, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def dice_coefficient(self, pred, target):
        batch_size = pred.shape[0]
        p = pred[:, 1].contiguous().view(batch_size, -1)
        t = target.view(batch_size, -1)
        
        # |p and t| => TP
        intersection = (p*t).sum(dim=1)
        #       2*TP                          /    |p| U |t| => 2*TP + FN + FP
        return (2.*intersection +self.smooth) / (p.sum(dim=1)+t.sum(dim=1)+self.smooth)

    def forward(self, pred, targets):
        batch_size = targets.shape[0]

        # dice score
        dice_loss = self.dice_coefficient(pred, targets)
        dice_loss = (1 - dice_loss).sum()/batch_size

        return dice_loss

# test
if __name__ == '__main__':
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action="store_true", default=False, help='')
    args = parser.parse_args()

    if args.gpu:
        map_device = torch.device('cuda')
    else:
        map_device = torch.device('cpu')

    """
    outputs should be like this with following tensors
        * means not in mask class

        batch0 p: class0=0.5, class1=0.5, class2=1.0
        batch1 p: class0=0.0*, class1=0.0*, class2=0.0
        mean   p: class0=0.5, class1=0.5, class2=0.5

        batch0 j: class0=0.25, class1=0.4, class2=1.0
        batch1 j: class0=0.0*, class1=0.0*, class2=0.0
        mean   j: class0=0.25, class1=0.4, class2=0.5
    """
    input_tensor = [
                    [[0,1,1],
                     [0,2,2],
                     [1,1,2]],
                    [[1,1,1],
                     [0,1,0],
                     [0,0,0]]
                    ]
    gt_tensor = [
                    [[1,1,1],
                     [0,2,2],
                     [0,0,2]],
                    [[2,2,2],
                     [2,2,2],
                     [2,2,2]]
                ]

    p = torch.LongTensor(input_tensor).to(device=map_device)
    g = torch.LongTensor(gt_tensor).to(device=map_device)

    results = [
                "pixel accuracy",
                pixel_accuracy(p, g, map_device=map_device),
                pixel_accuracy(p, g, size_average=False, map_device=map_device),
                "precision",
                precision(p, g, class_num=3, size_average=True, map_device=map_device),
                precision(p, g, class_num=3, size_average=False, map_device=map_device),
                "jaccard index",
                jaccard_index(p, g, class_num=3, size_average=True, map_device=map_device),
                jaccard_index(p, g, class_num=3, size_average=False, map_device=map_device)
              ]

    print("prediction tensor")
    print(p)
    print("ground truth tensor")
    print(g)
    for result in results:
        print(result)

    # speed check
    p = torch.randint(0, 10, (16, 512, 512)).to(device=map_device, dtype=torch.long)
    g = torch.randint(0, 10, (16, 512, 512)).to(device=map_device, dtype=torch.long)
    start = time.time()
    results = [
                pixel_accuracy(p, g, map_device=map_device),
                precision(p, g, class_num=3, size_average=True, map_device=map_device),
                jaccard_index(p, g, class_num=3, size_average=True, map_device=map_device)
              ]
    elapsed_time = time.time() - start
    print ("elapsed_time:{} sec".format(elapsed_time))
