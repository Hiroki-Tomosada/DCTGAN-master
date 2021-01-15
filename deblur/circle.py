import numpy as np

def circle(r):
    output = np.ones((256, 256))
    for i in range(256):
        for j in range(256):
            R = i ** 2 + j ** 2
            if R < r ** 2:
                output[i,j] = 0 
    return output
