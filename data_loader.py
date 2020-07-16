import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class DataLoader():
    def __init__(self, path_to_geant4_images_file, path_to_delphes_images_file, img_res=(32, 32)):
        self.img_res = img_res

        self.geant4_images = pd.read_hdf(path_to_geant4_images_file).image
        self.delphes_images = pd.read_hdf(path_to_delphes_images_file).image

    def load_data(self, batch_size=1, is_testing=False):

        sample_images = random.sample(
            list(zip(self.geant4_images, self.delphes_images)), batch_size)

        imgs_geant4, imgs_delphes = [], []

        for img in sample_images:
            img_geant4 = scipy.misc.imresize(img[0], self.img_res)
            img_delphes = scipy.misc.imresize(img[1], self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_geant4 = np.fliplr(img_geant4)
                img_delphes = np.fliplr(img_delphes)

            imgs_geant4.append(img_geant4)
            imgs_delphes.append(img_delphes)

        # Standardize the images
        imgs_geant4 = np.array(imgs_geant4)
        #imgs_geant4 = imgs_geant4/(np.max(imgs_delphes)/2) - 1

        imgs_delphes = np.array(imgs_delphes)
        #imgs_delphes = imgs_delphes/(np.max(imgs_delphes)/2) - 1

        #imgs_geant4 = np.array(imgs_geant4)/127.5 - 1.
        #imgs_delphes = np.array(imgs_delphes)/127.5 - 1.


        imgs_geant4 = np.expand_dims(imgs_geant4, axis=-1)
        imgs_delphes = np.expand_dims(imgs_delphes, axis=-1)

        return imgs_geant4, imgs_delphes

    def load_batch(self, batch_size=1, is_testing=False):

        geant4_images  = self.geant4_images
        delphes_images = self.delphes_images

        self.n_batches = int(len(geant4_images) / batch_size)

        for i in range(self.n_batches-1):
            batch = list(zip(geant4_images[i*batch_size:(i+1)*batch_size], delphes_images[i*batch_size:(i+1)*batch_size]))

            imgs_geant4, imgs_delphes = [], []

            for img in batch:

                img_geant4 = scipy.misc.imresize(img[0], self.img_res)
                img_delphes = scipy.misc.imresize(img[1], self.img_res)

                #if not is_testing and np.random.random() > 0.5:
                #    img_geant4 = np.fliplr(img_geant4)
                #    img_delphes = np.fliplr(img_delphes)

                imgs_geant4.append(img_geant4)
                imgs_delphes.append(img_delphes)

            # Standardize the images
            imgs_geant4 = np.array(imgs_geant4)
            #imgs_geant4 = imgs_geant4/(np.max(imgs_geant4)/2) - 1

            imgs_delphes = np.array(imgs_delphes)
            #imgs_delphes = imgs_delphes/(np.max(imgs_delphes)/2) - 1

            #imgs_geant4 = np.array(imgs_geant4)/127.5 - 1.
            #imgs_delphes = np.array(imgs_delphes)/127.5 - 1.

            imgs_geant4 = np.expand_dims(imgs_geant4, axis=-1)
            imgs_delphes = np.expand_dims(imgs_delphes, axis=-1)

            yield imgs_geant4, imgs_delphes

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
