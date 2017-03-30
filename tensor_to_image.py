import numpy as np
def tensor_to_image(tensor):
    tensor = 0.1*((tensor - tensor.mean())/(tensor.std() + 1e-6))
    tensor = np.clip(tensor+0.5,0,1)
    tensor *= 255
    return np.clip(tensor.transpose(1,2,0),0,255).astype('uint8')