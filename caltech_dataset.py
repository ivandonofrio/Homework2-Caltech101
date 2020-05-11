from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):

    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class)
        '''

        # Importing each image from provided files
        with open(f'./Caltech101/{self.split}.txt', 'r') as split_file:

            # Get all split lines
            lines = split_file.read().split('\n')

            # Create an image dictionary
            # Store each image as (class, image) pair
            self.images = {}
            self.labels = {}
            self.indexes = {}
            for index, line in enumerate([line for line in lines if line and 'BACKGROUND' not in line]):

                # Convert label to integer
                label = line.split('/')[0]
                if label not in self.labels:

                    encoding = len(self.labels)
                    self.labels[label] = encoding
                    self.indexes[encoding] = label

                # Store new image in proper dectionary
                self.images[index] = (pil_loader(f'./Caltech101/101_ObjectCategories/{line}'), self.labels[label])


    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.images[index] # Provide a way to access image and label via index
                                          # Image should be a PIL Image
                                          # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.images) # Provide a way to get the length (number of elements) of the dataset
        return length

    def get_validation(self, stratify=False, validation_size=.5):

        # Get indexes and corresponding images and split them
        indexes = list(self.images.keys())
        labels = list(map(lambda image: image[0], self.images.values()))

        valid_split = False
        while not valid_split:

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(indexes, labels, test_size=validation_size, stratify=(labels if stratify is True else None))

            # Check the length of unique classes in each split
            train_all_classes = len(set(y_train))
            test_all_classes = len(set(y_test))

            # Evaluate exit condition
            valid_split = (train_all_classes == test_all_classes and train_all_classes == 101)

        # Get split indexes
        return X_train, X_test
