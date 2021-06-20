import nn
import time
class DigitClassificationModel(object):
    def __init__(self):
        self.batch_size = 10
        self.m1 = nn.Parameter(784,15)
        self.m2 = nn.Parameter(15,10)
        self.b1 = nn.Parameter(1,15)
        self.b2 = nn.Parameter(1,10)
    def run(self, x):
        xm = nn.Linear(x,self.m1)
        a1 = nn.AddBias(xm, self.b1)
        a2 = nn.ReLU(a1)
        a3 = nn.Linear(a2,self.m2)
        a4 = nn.AddBias(a3, self.b2)
        return a4
    def get_loss(self, x, y):
        answer = self.run(x)
        return nn.SoftmaxLoss(answer, y)
    def train(self, dataset):
        multiplier=-0.04
        for i in range(10000):
            sample = 0
            total_loss = 0
            for x,y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                #print(nn.as_scalar(loss))
                grad_wrt_m1,grad_wrt_b1,grad_wrt_m2,grad_wrt_b2=nn.gradients(loss,[self.m1,self.b1,self.m2,self.b2])
                self.m1.update(grad_wrt_m1, multiplier)
                self.m2.update(grad_wrt_m2, multiplier)
                self.b1.update(grad_wrt_b1, multiplier)
                self.b2.update(grad_wrt_b2, multiplier)
                total_loss = total_loss+nn.as_scalar(loss)*self.batch_size
                sample = sample+self.batch_size
            temp = dataset.get_validation_accuracy()
            #print(temp)
            if(temp>0.973):
                break
