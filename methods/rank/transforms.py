class ToRGB(object):

    def __call__(self, tensor):

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        
        if tensor.shape[0] == 1:
            tensor = tensor.expand(3, -1, -1)

        return tensor