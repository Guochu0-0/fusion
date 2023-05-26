# coding: utf-8
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))

        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform

    def __getitem__(self, index):
        vi_path, ir_path = self.imgs[index]
        vi_img = transforms.Grayscale()(Image.open(vi_path))
        ir_img = transforms.Grayscale()(Image.open(ir_path))
        vi_img = transforms.ToTensor()(vi_img)
        ir_img = transforms.ToTensor()(ir_img)
        cat_img = torch.cat([vi_img, ir_img], 0)
        cat_img = transforms.RandomCrop(120)(cat_img)
        vi_img = torch.unsqueeze(cat_img[0], 0)
        ir_img = torch.unsqueeze(cat_img[1], 0)
        return vi_img, ir_img

    def __len__(self):
        return len(self.imgs)

