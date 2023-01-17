from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import utils
from .tsv import TSVDataset
from .tsv_openimage import TSVOpenImageDataset

from .samplers import DistributedChunkSampler
from .comm import comm
from pathlib import Path
import torchvision.transforms.functional as F
from torchvision.transforms.transforms import RandomResizedCrop, Compose, RandomHorizontalFlip


class CenterCropWithPos(RandomResizedCrop):
    max_size = 10000
    x = torch.arange(max_size).repeat(max_size, 1)[None, :]
    _pos = torch.cat((x, x.permute(0, 2, 1)), dim=0).float()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transforms.CenterCrop(224)

    def forward(self, img):
        # PIL convention
        w_pil, h_pil = img.size
        pos = RandomResizedCropWithPos._pos[:, :h_pil, :w_pil]
        out = self.transform(img)
        out_pos = self.transform(pos)
        return out, out_pos

def build_dataloader(args, is_train=True):
    assert(args.aug_opt == 'dino_aug')
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.local_crops_size,
    )

    if 'imagenet1k' in args.dataset:
        if args.zip_mode:
            from .zipdata import ZipData
            if is_train:
                datapath = os.path.join(args.data_path, 'train.zip')
                data_map = os.path.join(args.data_path, 'train_map.txt')

            dataset = ZipData(
                datapath, data_map,
                transform
            )
        elif args.tsv_mode:
            map_file = None
            dataset = TSVDataset(
                os.path.join(args.data_path, 'train.tsv'),
                transform=transform,
                map_file=map_file
            )
        else:
            #####################################################
            import time
            import tarfile
            if len(args.untar_path) > 0 and args.untar_path[0] == '$':
                args.untar_path = os.environ[args.untar_path[1:]]

            start_copy_time = time.time()
            if args.data_path.split('/')[-1].split('.')[-1] == 'ilsvrc2012.tar':
                if int(args.gpu) == 0:
                    with tarfile.open(args.data_path, 'r') as f:
                        f.extractall(args.untar_path)

                    print('Time taken for untar:', time.time() - start_copy_time)
                    print(os.listdir(args.untar_path))

                args.data_path = os.path.join(args.untar_path, args.data_path.split('/')[-1].split('.')[0],
                                              'ILSVRC2012_img_train')
                args.data_path_val = os.path.join(args.untar_path, args.data_path.split('/')[-1].split('.')[0],
                                                  'ILSVRC2012_img_val')
            torch.distributed.barrier()

            ######################################################
            dataset = datasets.ImageFolder(args.data_path, transform=transform)
    elif 'imagenet22k' in args.dataset:
        dataset = _build_vis_dataset(args, transforms=transform, is_train=True)
    elif 'webvision1' in args.dataset:
        dataset = webvision_dataset(args, transform=transform, is_train=True)
    elif 'openimages_v4' in args.dataset:
        dataset = _build_openimage_dataset(args, transforms=transform, is_train=True)

    else:
        # only support folder format for other datasets
        dataset = datasets.ImageFolder(args.data_path, transform=transform)

    if args.sampler == 'distributed':
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    elif args.sampler == 'chunk':
        chunk_sizes = dataset.get_chunk_sizes() \
            if hasattr(dataset, 'get_chunk_sizes') else None
        sampler = DistributedChunkSampler(
            dataset, shuffle=True, chunk_sizes=chunk_sizes
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    return data_loader



def _build_vis_dataset(args, transforms, is_train=True):
    if comm.is_main_process():
        phase = 'train' if is_train else 'test'
        print('{} transforms: {}'.format(phase, transforms))

    dataset_name = 'train' if is_train else 'val'
    if args.tsv_mode:

        if args.dataset == 'imagenet22k':
            map_file = os.path.join(args.data_path, 'labelmap_22k_reorder.txt')
        else:
            map_file = None

        if os.path.isfile(os.path.join(args.data_path, dataset_name + '.tsv')):
            tsv_path = os.path.join(args.data_path, dataset_name + '.tsv')
        elif os.path.isdir(os.path.join(args.data_path, dataset_name)):
            tsv_list = []
            if len(tsv_list) > 0:
                tsv_path = [
                    os.path.join(args.data_path, dataset_name, f)
                    for f in tsv_list
                ]
            else:
                data_path = os.path.join(args.data_path, dataset_name)
                tsv_path = [
                    str(path)
                    for path in Path(data_path).glob('*.tsv')
                ]
            logging.info("Found %d tsv file(s) to load.", len(tsv_path))
        else:
            raise ValueError('Invalid TSVDataset format: {}'.format(args.dataset ))

        sas_token_file = [
            x for x in args.data_path.split('/') if x != ""
        ][-1] + '.txt'

        if not os.path.isfile(sas_token_file):
            sas_token_file = None
        logging.info("=> SAS token path: %s", sas_token_file)

        dataset = TSVDataset(
            tsv_path,
            transform=transforms,
            map_file=map_file,
            token_file=sas_token_file
        )
    else:
        dataset = datasets.ImageFolder(args.data_path, transform=transforms)
    print("%s set size: %d", 'train' if is_train else 'val', len(dataset))

    return dataset




class webvision_dataset(Dataset): 
    def __init__(self, args, transform, num_class=1000, is_train=True): 
        self.root = args.data_path
        self.transform = transform

        self.train_imgs = []
        self.train_labels = {}    
             
        with open(os.path.join(self.root, 'info/train_filelist_google.txt')) as f:
            lines=f.readlines()    
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.train_imgs.append(img)
                    self.train_labels[img]=target            
        
        with open(os.path.join(self.root, 'info/train_filelist_flickr.txt')) as f:
            lines=f.readlines()    
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.train_imgs.append(img)
                    self.train_labels[img]=target            

    def __getitem__(self, index):

        img_path = self.train_imgs[index]
        target = self.train_labels[img_path]
        file_path = os.path.join(self.root, img_path)

        image = Image.open(file_path).convert('RGB')   
        img = self.transform(image)        

        return img, target
                

    def __len__(self):
        return len(self.train_imgs)



def _build_openimage_dataset(args, transforms, is_train=True):

    files = 'train.tsv:train.balance_min1000.lineidx:train.label.verify_20191102.tsv:train.label.verify_20191102.6962.tag.labelmap'
    items = files.split(':')
    assert len(items) == 4, 'openimage dataset format: tsv_file:lineidx_file:label_file:map_file'

    root = args.data_path
    dataset = TSVOpenImageDataset(
        tsv_file=os.path.join(root, items[0]),
        lineidx_file=os.path.join(root, items[1]),
        label_file=os.path.join(root, items[2]),
        map_file=os.path.join(root, items[3]),
        transform=transforms
    )

    return dataset


class RandomResizedCropWithPos(RandomResizedCrop):
    max_size = 10000
    x = torch.arange(max_size).repeat(max_size, 1)[None, :]
    _pos = torch.cat((x, x.permute(0, 2, 1)), dim=0).float()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        # PIL convention
        w_pil, h_pil = img.size
        pos = RandomResizedCropWithPos._pos[:, :h_pil, :w_pil]
        out = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        out_pos = F.resized_crop(pos, i, j, h, w, self.size, self.interpolation)
        return out, out_pos


class MyCompose(Compose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        flip_bool = 0
        pos = None
        for t in self.transforms:
            if type(t) == RandomResizedCropWithPos or type(t) == CenterCropWithPos:
                img, pos = t(img)
            elif type(t) == MyComposeInner:
                img, flip_bool = t(img)
            else:
                img = t(img)
        if flip_bool == 1:
            return img, F.hflip(pos)
        return img, pos


class MyComposeInner(Compose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        flip_bool = 0
        for t in self.transforms:
            if type(t) == RandomHorizontalFlipWithFlipBool:
                img, flip_bool = t(img)
            else:
                img = t(img)
        return img, flip_bool


class RandomHorizontalFlipWithFlipBool(RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img):
        if torch.rand(1) < self.p:
            return F.hflip(img), 1
        return img, 0


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, local_crops_size=96):
        flip_and_color_jitter = MyComposeInner([
            RandomHorizontalFlipWithFlipBool(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = MyCompose([
            RandomResizedCropWithPos(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = MyCompose([
            RandomResizedCropWithPos(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        if not isinstance(local_crops_size, tuple) or not isinstance(local_crops_size, list):
            local_crops_size = list(local_crops_size)

        
        if not isinstance(local_crops_number, tuple) or not isinstance(local_crops_number, list):
            local_crops_number = list(local_crops_number)

        self.local_crops_number = local_crops_number

        self.local_transfo = []
        for l_size in local_crops_size:
            self.local_transfo.append(MyCompose([
                RandomResizedCropWithPos(l_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ]))

    def __call__(self, image):
        crop1, pos1 = self.global_transfo1(image)
        crop2, pos2 = self.global_transfo2(image)
        crops = [crop1, crop2]
        crops_pos = [pos1, pos2]
        # print(f'self.local_crops_number {self.local_crops_number}')
        for i, n_crop in enumerate(self.local_crops_number):
            for _ in range(n_crop):
                c, p = self.local_transfo[i](image)
                crops.append(c)
                crops_pos.append(p)
        return crops, crops_pos
