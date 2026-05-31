from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, save_path):
            with open(save_path, 'rb') as f:
                data_list = pickle.load(f)

            size_list = data_list[0]
            act_func = data_list[1]
            self.__init__(size_list=size_list, act_func=act_func)           
            param_list = data_list[2:] 
            param_id = 0        
            for layer in self.layers:
                if layer.optimizable:
                    if param_id < len(param_list):
                        current_params = param_list[param_id]
                        layer.params['W'][...] = current_params['W']
                        layer.params['b'][...] = current_params['b']
                        layer.weight_decay = current_params['weight_decay']
                        layer.weight_decay_lambda = current_params['lambda']
                        param_id += 1
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def clear_grad(self):
        for layer in self.layers:
            if layer.optimizable:
                layer.clear_grad()
        

import numpy as np
import pickle
from .op import conv2D, Linear, ReLU   

class Model_CNN(Layer):

    def __init__(self,conv_params_list,fc_dims,activation='ReLU'):
        super().__init__()

        self.conv_params_list = conv_params_list
        self.fc_dims = fc_dims
        self.activation = activation
        self.optimizable = True
        self.conv_layers = []
        self.fc_layers = []
        self.layers = []
        self.conv_out_shape = None
        self.flat_size = None
        self.fc_built = False

        for params in conv_params_list:
            conv = conv2D(in_channels=params['in_channels'],out_channels=params['out_channels'],kernel_size=params['kernel_size'],stride=params['stride'],padding=params['padding'])
            self.conv_layers.append(conv)
            self.layers.append(conv)
            if activation == 'ReLU':
                relu = ReLU()
                self.conv_layers.append(relu)
                self.layers.append(relu)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.conv_layers:
            outputs = layer(outputs)
        self.conv_out_shape = outputs.shape
        N = outputs.shape[0]
        flat = outputs.reshape(N, -1)
        self.flat_size = flat.shape[1]
        if not self.fc_built:
            self._build_fc_layers(self.flat_size)
            self.fc_built = True
        outputs = flat
        for layer in self.fc_layers:
            outputs = layer(outputs)
        return outputs

    def _build_fc_layers(self, input_dim):

        dims = [input_dim] + self.fc_dims
        for i in range(len(dims) - 1):
            linear = Linear(in_dim=dims[i],out_dim=dims[i + 1])
            self.fc_layers.append(linear)
            self.layers.append(linear)
            if i < len(dims) - 2 and self.activation == 'ReLU':
                relu = ReLU()
                self.fc_layers.append(relu)
                self.layers.append(relu)

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.fc_layers):
            grad = layer.backward(grad)
        grad = grad.reshape(self.conv_out_shape)
        for layer in reversed(self.conv_layers):
            grad = layer.backward(grad)
        return grad

    def save_model(self, save_path):
        state_dict = []
        for layer in self.layers:
            if layer.optimizable:
                state_dict.append({'params': {k: v.copy()for k, v in layer.params.items()}})
        save_dict = {'conv_params_list': self.conv_params_list,'fc_dims': self.fc_dims,'activation': self.activation,'state_dict': state_dict}
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load_model(self, save_path):
        with open(save_path, 'rb') as f:
            save_dict = pickle.load(f)

        self.__init__(save_dict['conv_params_list'], save_dict['fc_dims'], save_dict['activation'])
        dummy_X = np.zeros((1, 1, 28, 28), dtype=np.float32)
        self.forward(dummy_X)
        state_dict = save_dict['state_dict']
        param_id = 0
        
        for layer in self.layers:
            if layer.optimizable:
                for key in layer.params:
                    if param_id < len(state_dict):
                        layer.params[key][...] = state_dict[param_id]['params'][key]
                param_id += 1

    def clear_grad(self):
        for layer in self.layers:
            if layer.optimizable:
                layer.clear_grad()