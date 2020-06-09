import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class COCODataset(Dataset):
    """
    COCODataset is for Pytorch DataLoder:
    it reads the prepocessed file and provide train , Validation and test dataset. 
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        input:
        -----
        data_folder: path of folder contains preprocessed image in h5py format and captions 
        data_name: base name of processed datasets
        split:  'TRAIN', 'VAL', or 'TEST'
        transform: image transform pipeline
        """
        self.split = split

        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # read hdf5_file
        hdf5_file = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')

        self.images = hdf5_file['images']

        # Captions per image
        self.cpi = hdf5_file.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)


    def __getitem__(self, i):

        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.images[i // self.cpi] / 255.)
        if self.transform:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score

            start_index = ((i // self.cpi) * self.cpi)
            end_index = start_index +self.cpi
            all_captions = torch.LongTensor(self.captions[start_index:end_index])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size