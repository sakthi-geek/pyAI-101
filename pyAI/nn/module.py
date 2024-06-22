import pickle


class Module:
    def __init__(self):
        self.parameters = []

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def add_parameter(self, param):
        if isinstance(param, Module):
            self.parameters += param.parameters
        else:
            self.parameters.append(param)
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump([param.data for param in self.parameters], f)

    def load(self, path):
        with open(path, 'rb') as f:
            param_data_list = pickle.load(f)
        for param, data in zip(self.parameters, param_data_list):
            param.data = data


