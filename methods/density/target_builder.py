import numpy as np

from math import floor
from scipy.signal import gaussian
from skimage.filters import gaussian as gaussian_filter

    
class DensityTargetBuilder:
    """ This builds the density map for counting. """

    def __init__(self, k_size=51, sigma=30, method='reflect', target_normalize_scale_factor=1.0, **kwargs):
        """ Constructor.
        Args:
            kernel_size (int, optional): Size (in px) of the kernel of the gaussian localizing an object. Defaults to 51.
            sigma (int, optional): Sigma of the gaussian. Defaults to 30.
        """

        assert method in ('move', 'reflect', 'normalize', 'move-cv2', 'normalize-scale'), f'Unsupported method: {method}'

        self.kernel_size = k_size
        self.sigma = sigma
        self.method = method
        self.target_normalize_scale_factor=target_normalize_scale_factor

    def build(self, shape, locations):
        if self.method == 'move':
            method = self.build_nocv2
        elif self.method == 'move-cv2':
            method = self.build_cv2
        elif self.method == 'reflect':
            method = self.build_reflect
        elif self.method == 'normalize':
            method = self.build_normalize
        
        return method(shape, locations)
    
    def build_cv2(self, shape, locations):
        import cv2
        """ This builds the density map, putting a gaussian over each dots localizing an object. """

        kernel_size = self.kernel_size
        sigma = self.sigma

        dmap = np.zeros(shape, dtype=np.float32)

        if len(locations) == 0:  # empty patch
            return dmap

        for i, center in enumerate(locations):
            H = np.multiply(cv2.getGaussianKernel(kernel_size, sigma),
                            (cv2.getGaussianKernel(kernel_size, sigma)).T)

            x = min(shape[1], max(1, abs(int(floor(center[1])))))
            y = min(shape[0], max(1, abs(int(floor(center[0])))))

            if x > shape[1] or y > shape[0]:
                continue

            x1 = x - int(floor(kernel_size / 2))
            y1 = y - int(floor(kernel_size / 2))
            x2 = x + int(floor(kernel_size / 2))
            y2 = y + int(floor(kernel_size / 2))
            dfx1 = 0
            dfy1 = 0
            dfx2 = 0
            dfy2 = 0
            change_H = False

            if x1 < 0:
                dfx1 = abs(x1)
                x1 = 0
                change_H = True
            if y1 < 0:
                dfy1 = abs(y1)
                y1 = 0
                change_H = True
            if x2 > shape[1] - 1:
                dfx2 = x2 - (shape[1] - 1)
                x2 = shape[1] - 1
                change_H = True
            if y2 > shape[0] - 1:
                dfy2 = y2 - (shape[0] - 1)
                y2 = shape[0] - 1
                change_H = True

            x1h = 1 + dfx1
            y1h = 1 + dfy1
            x2h = kernel_size - dfx2
            y2h = kernel_size - dfy2
            if change_H is True:
                H = np.multiply(cv2.getGaussianKernel(int(y2h - y1h + 1), sigma),
                                (cv2.getGaussianKernel(int(x2h - x1h + 1), sigma)).T)  # H.shape == (r, c)

            dmap[y1: y2 + 1, x1: x2 + 1] = dmap[y1: y2 + 1, x1: x2 + 1] + H
            
        dmap *= self.target_normalize_scale_factor
        return dmap

    def build_nocv2(self, hw, points_yx):
        """ This builds the density map, putting a gaussian over each dots localizing a perineural net.

            NOTE: This is an equivalent cv2-free implementation, but the results are NOT numerically identical to build_cv2().
            This is due to cv2.getGaussianKernel() and scipy.signal.gaussian() giving different results when
            the kernel size is even (according to cv2's doc, only odd kernel should be used).
            However, this difference is negligible in practice.
        """
        dmap = np.zeros(hw, dtype=np.float32)

        r = self.kernel_size // 2
        centers = np.abs(np.floor(points_yx).astype(int))
        for c_yx in centers:
            c_yx = np.clip(c_yx, 1, hw)
            lt = np.clip(c_yx - r, 0, None)
            rb = np.clip(c_yx + r + 1, None, hw)
            rh, rw = rb - lt

            kh = gaussian(rh, self.sigma)
            kw = gaussian(rw, self.sigma)
            kh /= kh.sum()
            kw /= kw.sum()

            H = np.outer(kh, kw)

            (y0, x0), (y1, x1) = lt, rb
            dmap[y0:y1, x0:x1] += H

        dmap *= self.target_normalize_scale_factor
        return dmap

    def build_reflect(self, hw, points_yx):
        """ This builds the density map, putting a gaussian over each dots localizing an object.
            It deals with borders by reflecting outside density inside the region.
        """
        r = self.kernel_size // 2
        padded_hw = np.array(hw) + 2 * r
        dmap = np.zeros(padded_hw, dtype=np.float32)

        centers = np.abs(np.floor(points_yx).astype(int))
        for c_yx in centers:
            c_yx = np.clip(c_yx, 0, hw) + r
            tl = c_yx - r
            br = c_yx + r
            rh, rw = br - tl

            kh = gaussian(rh, self.sigma)
            kw = gaussian(rw, self.sigma)
            kh /= kh.sum()
            kw /= kw.sum()

            H = np.outer(kh, kw)

            (y0, x0), (y1, x1) = tl, br
            dmap[y0:y1, x0:x1] += H
        
        ### reflect pad areas into the 'inside'
        # LEFT & RIGHT
        dmap[:, r:2*r] += dmap[:, :r][:, ::-1]
        dmap[:, -2*r:-r] += dmap[:, -r:][:, ::-1]

        # set to zero to avoid readding corners
        dmap[:, :r] = 0 
        dmap[:, -r:] = 0

        # TOP & BOTTOM
        dmap[r:2*r, :] += dmap[:r, :][::-1, :]
        dmap[-2*r:-r, :] += dmap[-r:, :][::-1, :]

        # unpad
        dmap = dmap[r:-r, r:-r]
        dmap *= self.target_normalize_scale_factor
        return dmap

    def build_normalize(self, hw, points_yx):
        num_points = len(points_yx)
        density_map = np.zeros(hw, dtype=np.float32)

        if num_points == 0:
            return density_map

        points_yx = np.clip(np.rint(points_yx), 0, np.array(hw) - 1).astype(int)
        for r, c in points_yx:
            density_map[r, c] += 1
        
        density_map = gaussian_filter(density_map, sigma=self.sigma, mode='constant')
        density_map = num_points * density_map / density_map.sum()  # renormalize density map
        density_map *= self.target_normalize_scale_factor
        return density_map

    def pack(self, image, target, pad=None):
        dmap = np.expand_dims(target, axis=-1)
        dmap = np.pad(dmap, pad) if pad else dmap
        # stack in a unique RGB-like tensor, useful for applying data augmentation
        return np.concatenate((image, dmap), axis=-1)
