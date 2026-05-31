from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if initialize_method is None:
            # He Initialization for Linear
            initialize_method = lambda size: np.random.normal(0, np.sqrt(2.0 / size[0]), size)
            self.W = initialize_method((in_dim, out_dim))
            self.b = initialize_method((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params["W"] + self.params["b"]

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads["W"] = self.input.T @ grad
        self.grads["b"] = np.sum(grad, axis=0, keepdims=True)
        output = grad @ self.params["W"].T
        return output

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    2D Convolution Layer
    Input:  (N, Cin, H, W)
    Weight: (Cout, Cin, KH, KW)
    Output: (N, Cout, Hout, Wout)
    """
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,initialize_method=None,weight_decay=False,weight_decay_lambda=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        KH, KW = self.kernel_size
        if initialize_method is None:
            fan_in = in_channels * KW * KH
            initialize_method = lambda size: np.random.normal(0, np.sqrt(2.0 / fan_in), size)

        self.params = {
            'W': initialize_method((out_channels, in_channels, KH, KW)),
            'b': initialize_method((out_channels,))
        }
        self.grads = {'W': None, 'b': None}
        self.X = None
        self.X_padded = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        N, Cin, H, W = X.shape
        KH, KW = self.kernel_size
        P = self.padding
        S = self.stride
        H_out = (H + 2 * P - KH) // S + 1
        W_out = (W + 2 * P - KW) // S + 1

        X_padded = np.pad(X, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
        self.X_padded = X_padded

        Y = np.zeros((N, self.out_channels, H_out, W_out))
        W = self.params['W']
        b = self.params['b']

        for co in range(self.out_channels):
            kernel = W[co]          # (Cin, KH, KW)
            bias = b[co]
            for i in range(H_out):
                hs = i * S
                he = hs + KH
                for j in range(W_out):
                    ws = j * S
                    we = ws + KW
                    window = X_padded[:, :, hs:he, ws:we]   # (N, Cin, KH, KW)
                    Y[:, co, i, j] = np.sum(window * kernel, axis=(1,2,3)) + bias
        return Y

    def backward(self, grads):
        N, Cout, H_out, W_out = grads.shape
        KH, KW = self.kernel_size
        P = self.padding
        S = self.stride
        X_padded = self.X_padded
        W = self.params['W']
        dW = np.zeros_like(W)
        db = np.sum(grads, axis=(0,2,3))
        dX_padded = np.zeros_like(X_padded)

        for co in range(Cout):
            kernel = W[co]
            for i in range(H_out):
                hs = i * S
                he = hs + KH
                for j in range(W_out):
                    ws = j * S
                    we = ws + KW
                    g = grads[:, co, i, j]          # (N,)
                    window = X_padded[:, :, hs:he, ws:we]   # (N, Cin, KH, KW)
                    # dW
                    dW[co] += np.sum(g[:, None, None, None] * window, axis=0)
                    # dX
                    dX_padded[:, :, hs:he, ws:we] += g[:, None, None, None] * kernel[None, :, :, :]

        if self.weight_decay:
            dW += self.weight_decay_lambda * W

        if P > 0:
            dX = dX_padded[:, :, P:-P, P:-P]
        else:
            dX = dX_padded

        self.grads['W'] = dW
        self.grads['b'] = db
        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.input = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        if self.has_softmax:
            predicts = softmax(predicts)
        self.input = (predicts,labels)
        batch_size = predicts.shape[0]
        loss = 0
        for i in range(batch_size):
            loss += -np.log(predicts[i, labels[i]] + 1e-12) / batch_size
        return loss
        
    def backward(self):
        # first compute the grads from the loss to the input
        predicts, labels = self.input
        batch_size = predicts.shape[0]
        if self.has_softmax:
            one_hot = np.zeros_like(predicts)
            one_hot[np.arange(batch_size), labels] = 1
            self.grads = (predicts - one_hot) / batch_size
        else:
            self.grads = np.zeros_like(predicts)
            self.grads[np.arange(batch_size), labels] = -1 / (predicts[np.arange(batch_size), labels] + 1e-12) / batch_size
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


