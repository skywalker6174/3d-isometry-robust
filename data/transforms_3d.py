import numpy as np
import torch



class compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pointcloud):
        for t in self.transforms:
            pointcloud = t(pointcloud)
        return pointcloud

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class to_tensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud).float()


class normalize(object):
    def __init__(self, centroid = None, scale = None):
        self.centroid = centroid #(x,y,z) 1*3
        self.scale = scale # sclar, the maximal dist from centroid
    def __call__(self, pointcloud):
        # pointcloud N*3
        if self.centroid is None:
            self.centroid = np.mean(pointcloud, axis=0)
        pointcloud = pointcloud - self.centroid
        if self.scale is None:
            self.scale = np.max(np.sqrt(np.sum(pointcloud**2, axis=1)))
        pointcloud = pointcloud / self.scale
        return pointcloud

class rand_translate(object):
    def __init__(self, translate_range = 0.2):
        self.translate_range = translate_range
    def __call__(self, pointcloud):
        b = np.random.uniform(-self.translate_range, self.translate_range, size=[3])
        pointcloud = np.add(pointcloud, b).astype(np.float32)
        return pointcloud

class rand_scale(object):
    def __init__(self, low =2./3, high =3./2):
        self.low = low
        self.high = high
    def __call__(self, pointcloud):
        A = np.random.uniform(self.low, self.high, size=[3])
        pointcloud = np.multiply(pointcloud, A).astype(np.float32)
        return pointcloud

class jitter(object):
    def __init__(self, std = 0.01, clip = 0.02):
        self.std = std
        self.clip = clip
    def __call__(self, pointcloud):
        N, C = pointcloud.shape
        pointcloud += np.clip(self.std * np.random.randn(N, C), -1*self.clip, self.clip)
        return pointcloud

class rotate_y(object):
    def __init__(self, rotate_range = None):
        if rotate_range is None:
            self.rotate_range = [0, np.pi*2]
       
    def __call__(self, pointcloud):
        angle = np.random.uniform(low =self.rotate_range[0], high=self.rotate_range[1])
        cosv, sinv = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cosv,0, -sinv],
                                    [0 , 1, 0],
                                    [sinv,0, cosv]])
        pointcloud = np.dot(pointcloud, rotation_matrix).astype(np.float32)
        return pointcloud

class rotate_all(object):
    def __init__(self, p=0.5, rotate_range = None):
        if rotate_range is None:
            self.rotate_range = [0, np.pi*2]
        self.p = p
        
    def __call__(self, pointcloud):
        if np.random.uniform() < self.p:
            angles = np.random.uniform(low = self.rotate_range[0], high = self.rotate_range[1], size=3)
            cos1, sin1 = np.cos(angles[0]), np.sin(angles[0])
            cos2, sin2 = np.cos(angles[1]), np.sin(angles[1])
            cos3, sin3 = np.cos(angles[2]), np.sin(angles[2])
            rotation_matrix = np.array([[cos1*cos3-cos2*sin1*sin3, -cos2*cos3*sin1-cos1*sin3,  sin1*sin2],
                                        [cos3*sin1+cos1*cos2*sin3,  cos1*cos2*cos3-sin1*sin3, -cos1*sin2],
                                        [      sin2*sin3         ,       cos3*sin2          ,    cos2   ]])

            pointcloud = np.dot(pointcloud, rotation_matrix).astype(np.float32)
        return pointcloud

