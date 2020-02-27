import os
import h5py
import numpy as np
import glob
from torch.utils.data import Dataset
import json
from plyfile import PlyData, PlyElement

def modelnet40_download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'modelnet40_data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    modelnet40_download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'modelnet40_data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', transforms = None):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.transforms = transforms
        #self.cat = get_catogrey()
        #self.classes = list(self.cat.keys())
     

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        if self.transforms is not None:
            pointcloud = self.transforms(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class ShapeNetPart(Dataset):
    def __init__(self,
                 root='data/shape_data',
                 num_points=2500,
                 class_choice=None,
                 partition='train',
                 transforms = None):
        self.num_points = num_points
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.transforms = transforms
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(partition))
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        #print(self.classes)
       

    def __getitem__(self, index):
        fn = self.datapath[index]
        label = self.classes[self.datapath[index][0]]
        pointcloud = np.loadtxt(fn[1]).astype(np.float32)
        choice = np.random.choice(len(pointcloud), self.num_points, replace=True)
        #resample
        pointcloud = pointcloud[choice, :]

        if self.transforms is not None:
            pointcloud = self.transforms(pointcloud)
        pointcloud = pointcloud.astype('float32')
        label = np.array([label]).astype('int64')
    
        return pointcloud, label
        
    def __len__(self):
        return len(self.datapath)
  