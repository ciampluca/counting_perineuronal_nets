import ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np


if __name__ == "__main__":
    npImg1 = cv2.imread("einstein.png")

    img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
    img2 = torch.rand(img1.size())

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()


    img1 = Variable( img1,  requires_grad=False)
    img2 = Variable( img2, requires_grad = True)


    # Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
    ssim_value = ssim.ssim(img1, img2).item()
    print("Initial ssim:", ssim_value)

    # Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
    ssim_loss = ssim.SSIM()

    optimizer = optim.Adam([img2], lr=0.01)

    while ssim_value < 0.95:
        optimizer.zero_grad()
        ssim_out = -ssim_loss(img1, img2)
        print(ssim_out)
        ssim_value = - ssim_out.item()
        print(ssim_value)
        ssim_out.backward()
        optimizer.step()

