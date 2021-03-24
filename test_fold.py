import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image #Used to load a sample image


if __name__ == "__main__":
    B, C, W, H = 1, 3, 1024, 1024
    x = torch.randn(B, C, H, W)

    kernel_size = 128
    stride = 64
    patches = x.unfold(1, 3, 3).unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    print(patches.shape)

    output_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            for r in range(patches.shape[2]):
                for c in range(patches.shape[3]):
                    output_patches.append(patches[i, j, r, c, ...])
    output_patches = torch.stack(output_patches)
    rec_image = output_patches.view(patches.shape[0], patches.shape[2], patches.shape[3], *patches.size()[-3:])
    #rec_image = rec_image.permute(2, 0, 3, 1, 4).contiguous()
    rec_image = rec_image.permute(0, 3, 4, 5, 1, 2).contiguous()
    rec_image = rec_image.view(rec_image.shape[0], rec_image.shape[1], rec_image.shape[2] * rec_image.shape[3],
                               rec_image.shape[4] * rec_image.shape[5])
    rec_image = rec_image.view(rec_image.shape[0], rec_image.shape[1] * rec_image.shape[2], -1)

    output = F.fold(
        rec_image, output_size=(H, W), kernel_size=kernel_size, stride=stride)


    print("End")
