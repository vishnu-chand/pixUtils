import numpy as np
from pixUtils import *


def matmul(x, y):
    '''
    i = 2, 7, 5, 15, 31, 99
    j = 2, 7, 5, 15, 99, 31
    a = np.random.random(np.arange(np.product(i)).reshape(i).shape)
    c = np.random.random(np.arange(np.product(j)).reshape(j).shape)
    res2 = np.matmul(a, c)
    res = matmul(a, c)
    print((res2 - res).sum())
    '''
    if len(x.shape) == 2:
        return np.dot(x, y)
    else:
        return np.array([matmul(x, y) for x, y in zip(x, y)])


def norm(x=None):
    """
    x = [[0.1400, 0.3100, 0.4600, 0.2000, 0.6700],
        [0.3600, 0.6700, 0.3400, 0.0200, 0.7500],
        [0.1800, 0.5200, 0.9600, 0.2300, 0.4800],
        [0.6100, 0.3300, 0.0000, 0.3100, 0.2100],
        [0.8400, 0.9400, 0.7800, 0.2300, 0.5300]]
    """
    if x is None:
        x = torch.randint(0, 100, (5, 5)) / 100
    l2 = torch.norm(x, dim=1, keepdim=True)
    print([(i @ i).sum() ** .5 for i in x])
    print(l2.numpy().tolist())
    print("17 norm npImplementation : ", );
    quit()
    print([(i * i.T).sum() ** .5 for i in x])
    print([(i.dot(i)).sum() ** .5 for i in x])
    res = []
    for i in x:
        j = [j * j for j in i]
        res.append(sum(j) ** .5)
    print(res)


def softmaxTorch(x, axis=1, returnLogSoftmax=False):
    xMax = x.max(axis, keepdims=True)[0]
    tmp = x - xMax
    s = torch.exp(tmp).sum(axis, keepdim=True)
    out = tmp - np.log(s)
    if not returnLogSoftmax:
        out = torch.exp(out)
    return out


def softmax(x, axis, returnLogSoftmax=False):
    xMax = np.max(x, axis=axis, keepdims=True)
    tmp = x - xMax
    s = np.sum(np.exp(tmp), axis=axis, keepdims=True)
    out = tmp - np.log(s)
    if not returnLogSoftmax:
        out = np.exp(out)
    return out


def npNllLoss(yHat, y):
    ch = yHat.shape
    if len(ch) == 2:
        batch, width = ch
        nll = -yHat[range(0, batch), y].mean()
    elif len(ch) == 3:
        # implement copied from pytorch/aten/src/ATen/native/LossNLL2d.cpp
        yHat = yHat[:, :, None, :]
        y = y[:, None, :]
        batch, height, width = np.array(yHat.shape)[[0, 2, 3]]
        nll = 0
        c = 0
        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    cur_target = y[b][h][w]
                    nll += yHat[b][cur_target][h][w]
                    c += 1
        nll /= -c
    else:
        raise Exception("not implemented")
    return nll


def npCrossEntropy(yHat, y):
    yHat = softmax(yHat, 1, returnLogSoftmax=True)
    return npNllLoss(yHat, y)


def bce():
    """
    https://analyticsindiamag.com/all-pytorch-loss-function/
    """
    y_pred = np.array([0.1580, 0.4137, 0.2285])
    y_true = np.array([0.0, 1.0, 0.0])  # 2 labels: (0,1)

    def BCE(y_pred, y_true):
        total_bce_loss = np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
        # Getting the mean BCE loss
        num_of_samples = y_pred.shape[0]
        mean_bce_loss = total_bce_loss / num_of_samples
        return mean_bce_loss

    bce_value = BCE(y_pred, y_true)
    print("BCE error is: " + str(bce_value))

    bce_loss = torch.nn.BCELoss()
    input = torch.tensor(y_pred)
    target = torch.tensor(y_true)
    output = bce_loss(input, target)
    print("148 bce local output: ", output)


bce()
print("151  local : ", );
quit()


def temp():
    # setSeed(10)
    yHat = torch.randn(13, 53, 23)
    y = torch.randint(0, 5, (13, 23))
    from torch.nn.functional import cross_entropy
    a = cross_entropy(yHat, y)
    x = npCrossEntropy(yHat.numpy(), y.numpy())
    print("138 temp local a: ", a)
    print("120 temp local x: ", x)
    # print("121 temp local : ", s)


temp()
# temp()

"""

mkdir -p /home/hippo/awsBridge/virtualBGv4/db2 /home/hippo/awsBridge/v2; sudo ssh -i /home/hippo/awsBridge/.ssh/vishnu.pem ec2-user@ec2-54-197-1-23.compute-1.amazonaws.com 'mkdir -p /home/ec2-user/awsBridge;cd /home/ec2-user/virtualBGv4;tar -zcf /home/ec2-user/awsBridge/20_18Feb23_43215981.tar.gz db/deleteMeToo/temp/tboard'; sudo scp -i /home/hippo/awsBridge/.ssh/vishnu.pem -r ec2-user@ec2-54-197-1-23.compute-1.amazonaws.com:/home/ec2-user/awsBridge/20_18Feb23_43215981.tar.gz /home/hippo/awsBridge/v2; sudo ssh -i /home/hippo/awsBridge/.ssh/vishnu.pem ec2-user@ec2-54-197-1-23.compute-1.amazonaws.com 'rm -rf /home/ec2-user/awsBridge/20_18Feb23_43215981.tar.gz'; cd /home/hippo/awsBridge/virtualBGv4/db2;tar -zxf /home/hippo/awsBridge/v2/20_18Feb23_43215981.tar.gz; rm -rf /home/hippo/awsBridge/v2/20_18Feb23_43215981.tar.gz; echo -----------------------------------------; echo; echo; echo /home/hippo/awsBridge/virtualBGv4/db2/db/deleteMeToo/temp/tboard; echo; echo; echo; echo;


##### aws s3 cp /home/ec2-user/virtualBGv4/db/hrnetv0 s3://hippolms-stage-dev-storage/virtualBgBK/virtualBGv4/db/hrnetv0  --recursive
##### aws s3 cp /home/ec2-user/virtualBGv4/db/hrnetv0 s3://hippolms-stage-dev-storage/virtualBgBK/virtualBGv4/db/hrnetv0  --recursive


"""
