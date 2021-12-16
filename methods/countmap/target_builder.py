import numpy as np
from PIL import Image

    
class CountmapTargetBuilder:
    """ This builds the target to be used with the Count-Ception Network. """

    def __init__(self, scale=2, target_patch_size=32, stride=1, **kwargs):
        """ Constructor.
        Args:
            
        """
        
        self.scale = scale
        self.target_patch_size = target_patch_size
        self.stride = stride

    def build(self, shape, locations):
        framesize_h, framesize_w = int(shape[0] / self.scale), int(shape[1] / self.scale)
        label = self._build_label(locations, shape, framesize_h, framesize_w, self.stride, self.scale, self.target_patch_size)
        
        return label
    
    def _get_markers_cells(self, locations, scale, shape):
        size = int(shape[0] / scale), int(shape[1] / scale)
        
        label = np.zeros(shape)
        out = np.zeros(size)
        
        for point in locations:
            label[point[0], point[1]] = 1

        binsize = [scale, scale]
        for i in range(binsize[0]):
            for j in range(binsize[1]):
                out = np.maximum(label[i::binsize[0], j::binsize[1]], out)

        assert np.allclose(label.sum(), out.sum(), 1)

        return out
    
    def _get_cell_count_cells(self, markers, xyhw):
        x, y, h, w = xyhw

        return (markers[y:y + h, x:x + w] == 1).sum()
    
    def _get_labels_cells(self, markers, shape, scale, stride, patch_size, framesize_h, framesize_w):
        shape = int(shape[0] / scale), int(shape[1] / scale)
        height = int((int(shape[0]) + (patch_size // 2)*2) / stride)
        width = int((int(shape[1]) + (patch_size // 2)*2) / stride)

        label = np.zeros((height, width), dtype=np.float32)
        
        for y in range(0, height):
            for x in range(0, width):
                count = self._get_cell_count_cells(markers, (x * stride, y * stride, patch_size, patch_size))
                label[y][x] = count

        # count_total = self._get_cell_count_cells(markers, (0, 0, framesize_h + patch_size, framesize_w + patch_size))
        
        return label
    
    def _build_label(self, locations, shape, framesize_h, framesize_w, stride, scale, patch_size):
        base_x, base_y = 0, 0   # TODO modify if framesize != shape/scale
        markers = self._get_markers_cells(locations, scale, shape)
        markers = markers[base_y:base_y + framesize_h, base_x:base_x + framesize_w]
        markers = np.pad(markers, patch_size, "constant", constant_values=-1)

        label = self._get_labels_cells(markers, shape, scale, stride, patch_size, framesize_h, framesize_w)
        
        return label

    def pack(self, image, target, pad=None):
        label = np.pad(target, pad) if pad else target
        
        # eventually scale image as already done for the target
        image = np.array(Image.fromarray(image).resize((int(image.shape[0] / self.scale), int(image.shape[1] / self.scale))))
        # add pad to image to have same dimension of target
        image = np.pad(image, self.target_patch_size // 2)
        
        # stack in a unique RGB-like tensor, useful for applying data augmentation
        return np.stack((image, label), axis=-1)
