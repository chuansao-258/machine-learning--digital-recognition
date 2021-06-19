import nn
import time


class DigitClassificationModel(object):
    """
    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 10
        self.m1 = nn.Parameter(784,100)
        self.m2 = nn.Parameter(100,10)
        self.b1 = nn.Parameter(1,100)
        self.b2 = nn.Parameter(1,10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xm = nn.Linear(x,self.m1)
        a1 = nn.AddBias(xm, self.b1)
        a2 = nn.ReLU(a1)
        a3 = nn.Linear(a2,self.m2)
        a4 = nn.AddBias(a3, self.b2)
        return a4

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        answer = self.run(x)
        return nn.SoftmaxLoss(answer, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
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