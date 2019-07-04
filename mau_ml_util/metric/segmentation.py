"""
    Pytorch is expected.
"""

import torch

CPU = torch.device('cpu')
NAN = float('nan')

class SegmentationMetric(object):
    """
        the matrix is holding the value in order
                ground truth
                0 1 2 3 4 ...
        pred  0
              1
              2
              3
              4
              .
              .
              .

        so, in 3 class case with following tensor
        pred_tensor = [
                    [[0,1,1],
                     [0,2,2],
                     [1,1,2]],
                    ]
        gt_tensor = [
                    [[1,1,1],
                     [0,2,2],
                     [0,0,2]],
                    ]

        the matrix is goiing to be
               |ground truth
               |0 1 2
        -------+-----
        pred 0 |1 1 0
             1 |2 2 0
             2 |0 0 3

        and, to calc iou, for example,
        class 0:
            TP is 
               |0 1 2
            ---+-----
             0 |1 * *
             1 |* * *
             2 |* * *

            FP is 
               |0 1 2
            ---+-----
             0 |* 1 0
             1 |* * *
             2 |* * *

            FN is
               |0 1 2
            ---+-----
             0 |* * *
             1 |2 * *
             2 |0 * *

        so IoU of class 0 is TP/(TP+FP+FN) = 1/(1+1+2+) = 0.25

        and so on.
    """

    def __init__(self, class_num, map_device=CPU):
        self.class_num = class_num
        self.map_device = map_device

        self.class_matrix = torch.zeros(self.class_num, self.class_num).to(device=map_device, dtype=torch.long)

    def __call__(self, pred_labels, gt_labels):
        batch_size = pred_labels.shape[0]

        for batch in range(batch_size):
            self.__add_to_matrix(pred_labels[batch], gt_labels[batch])

    # per batch
    def __add_to_matrix(self, pred_label, gt_label):
        for gt_class_id in range(self.class_num):
            gt_class = torch.eq(gt_label, gt_class_id).to(dtype=torch.long)

            for pred_class_id in range(self.class_num):
                pred_class = torch.eq(pred_label, pred_class_id).to(dtype=torch.long)
                pred_class = torch.mul(gt_class, pred_class)
                count = torch.sum(pred_class)
                self.class_matrix[pred_class_id, gt_class_id] += count

    def calc_pix_acc(self):
        return float(torch.trace(self.class_matrix).cpu().item())/float(torch.sum(self.class_matrix).cpu().item())

    def calc_mean_pix_acc(self, ignore=[255]):
        if isinstance(ignore, int):
            ignore = [ignore]
        mean_pix_acc = {}

        for class_id in range(self.class_num):
            if class_id in ignore:
                continue

            all_class_id_pix = torch.sum(self.class_matrix[:, class_id]).cpu().item()
            if all_class_id_pix == 0:
                mean_pix_acc["class_{}".format(class_id)] = NAN
            else:
                mean_pix_acc["class_{}".format(class_id)] = (float(self.class_matrix[class_id, class_id].cpu().item())/float(all_class_id_pix))

        return mean_pix_acc

    def calc_mean_jaccard_index(self, ignore=[255]):
        """
            it is same to IoU
        """

        if isinstance(ignore, int):
            ignore = [ignore]
        iou = {}

        for class_id in range(self.class_num):
            if class_id in ignore:
                continue

            tpfpfn = (torch.sum(self.class_matrix[class_id, :])+torch.sum(self.class_matrix[:, class_id])-self.class_matrix[class_id, class_id]).cpu().item()
            if tpfpfn == 0:
                iou["class_{}".format(class_id)] = NAN
            else:
                iou["class_{}".format(class_id)] = (float(self.class_matrix[class_id, class_id].cpu().item())/float(tpfpfn))

        return iou

    def calc_mean_precision(self, ignore=[255]):
        if isinstance(ignore, int):
            ignore = [ignore]
        precision = {}

        for class_id in range(self.class_num):
            if class_id in ignore:
                continue

            tpfp = torch.sum(self.class_matrix[class_id, :]).cpu().item()
            if tpfp == 0:
                precision["class_{}".format(class_id)] = NAN
            else:
                precision["class_{}".format(class_id)] = (float(self.class_matrix[class_id, class_id].cpu().item())/float(tpfp))

        return precision

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
        following tensors outputs should be like this

        batch0 p: class0=0.5, class1=0.5, class2=1.0
        batch1 p: class0=0.0, class1=0.0, class2=0.0
        mean   p: class0=0.25, class1=0.25, class2=0.5

        batch0 j: class0=0.25, class1=0.4, class2=1.0
        batch1 j: class0=0.0, class1=0.0, class2=0.0
        mean   j: class0=0.125, class1=0.2, class2=0.5
    """
    pred_tensor = [
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

    p = torch.LongTensor(pred_tensor).to(device=map_device)
    g = torch.LongTensor(gt_tensor).to(device=map_device)

    print("prediction tensor")
    print(p)
    print("ground truth tensor")
    print(g)

    m = SegmentationMetric(3)
    m(p, g)

    print(m.class_matrix)
    print(m.calc_pix_acc())
    print(m.calc_mean_pix_acc())
    print(m.calc_mean_jaccard_index())
    print(m.calc_mean_precision())


    """
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
    """