import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--size', type=int, default=1000,
                    help='Generate size x size image')

args = parser.parse_args()

n=args.size

def get_iter(c:complex, thresh:int =4, max_steps:int =25) -> int:
    # Z_(n) = (Z_(n-1))^2 + c
    # Z_(0) = c
    z=c
    i=1
    while i<max_steps and (z*z.conjugate()).real<thresh:
        z=z*z +c
        i+=1
    return i

def plotter(n, thresh, max_steps=30):
    mx = 2.48 / (n-1)
    my = 2.26 / (n-1)
    mapper = lambda x,y: (mx*x - 2, my*y - 1.13)
    img=np.full((n,n), 255)
    for x in range(n):
        for y in range(n):
            it = get_iter(complex(*mapper(x,y)), thresh=thresh, max_steps=max_steps)
            img[y][x] = 255 - it
    return img

start = timer()
img = plotter(n, thresh=4, max_steps=100)
end = timer()
print('TIME: '+ str(end-start))
plt.imshow(img, cmap="plasma")
plt.axis("off")
plt.savefig('mandelbrot.png', format="png")