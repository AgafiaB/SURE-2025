from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torchvision.io import decode_image
from torch.utils.data import Dataset
import torch
import logging
import numpy as np
import cv2
from PIL import Image 
import torchvision.transforms as transforms
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.pairwise import nan_euclidean_distances
import numpy as np
import cv2
import skimage
from tqdm import tqdm
from itertools import combinations
import pandas as pd


class SQLDataset_Humanitarian(Dataset):
    '''
    Parameters: 
    - dir: The directory where the folder containing the images is located
    - file_path: The path to the TSV file containing the image paths and labels
    - transform: Optional transform to be applied on a sample (e.g., for data augmentation)
    '''
    def __init__(self, dir, file_path, transform=None):
        self.dir = Path(dir)
        self.file_path = file_path
        self.data = pd.read_csv(file_path, sep='\t')
        self.image_paths = self.data['image'].tolist()
        self.labels = self.data['label_image']
        self.label_dict = {value: key for key, value in 
                           self.data['label_image'].value_counts().reset_index()['label_image'].to_dict().items()}
        self.transform = transform
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        Returns a tuple of (image, label) for the given index.
        The image is loaded as a PIL Image, and the label is converted to an integer using the label_dict.
        If a transform was provided, it is applied to the image.
        '''
        img = Image.open(self.dir / self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        
        label = self.label_dict[label] # convert label to int
        return img, label
    
    def get_label_dict(self): 
        return self.label_dict

def powerset_without_emptyset(items):
    '''
    Returns the powerset of a list of items as a list of tuples, excluding the empty set
    '''
    combos = []
    for i in range(len(items)):
        combos.extend(list(combinations(items, len(items) - i)))
    return combos

def ensure_rgb(img):
    """
    Ensures the input image is a 3-channel RGB numpy array.
    If the image is grayscale (2D), it is converted to RGB.
    If the image is already RGB, it is returned unchanged.
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 2:  # Grayscale
            # CV2 has channels-last format
            assert len(img.expand_dims(axis=0).shape)==3, f"wrong shaped image: {img.shape}"
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).expand_dims(axis=0)
        elif img.ndim == 3 and img.shape[2] == 3:
            assert len(img.shape)==3, f"wrong shaped image: {img.shape}"
            return img  # Already RGB
        else:
            raise ValueError("Unsupported image shape for ensure_rgb: {}".format(img.shape))
    else:
        raise TypeError("Input must be a numpy ndarray.")
    

def apply_transformations(images, combo):
    """
    Applies a combination of transformations to a list of images.
    Returns a 2D array: (num_images, total_feature_dim)
    """
    all_feats = []
    for img in images:
        img_feats = []
        for t in combo:
            if t is cv2.HuMoments:
                im_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
                feat = cv2.normalize(cv2.HuMoments(cv2.moments(im_grey)).flatten(), None, 0, 255, cv2.NORM_MINMAX)
            elif t is skimage.feature.graycomatrix:
                im_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
                dists, angles = [1], [0]
                glcm = skimage.feature.graycomatrix(im_grey, distances=dists, angles=angles, symmetric=True, normed=True)
                diss = graycoprops(glcm, 'dissimilarity')
                contrast = graycoprops(glcm, 'contrast')
                correlation = graycoprops(glcm, 'correlation')
                energy = graycoprops(glcm, 'energy')
                homogeneity = graycoprops(glcm, 'homogeneity')
                cat = np.concatenate([diss, contrast, correlation, energy, homogeneity], axis=1)
                feat = cv2.normalize(cat, None, 0, 255, cv2.NORM_MINMAX).flatten()
            elif t is cv2.calcHist:
                feat = cv2.normalize(
                    cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256]*3).flatten(),
                    None, 0, 255, cv2.NORM_MINMAX).flatten()
            elif t is skimage.feature.local_binary_pattern:
                im_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
                lbp = skimage.feature.local_binary_pattern(im_grey, P=8, R=1)
                hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
                feat = hist.flatten()
            else:
                print(t)
                raise ValueError(f'Unsupported transformation: {t}')
            # flatten prior to concatenation 
            img_feats.append(feat.flatten())
        # Concatenate all features for this image into a feature vector for one image and append to all_feats
        all_feats.append(np.concatenate(img_feats))

    # returns (num_images, total_feature_dim), where each row corresponds to the concatenated features of one img
    return np.stack(all_feats)

def best_transformation(transformations, class1_imgs, class2_imgs):
    '''
    Parameters: 
        transformations - a list of transformation functions
        class1_imgs, class2_imgs - lists of ndarray images

    Returns:
        The transformation or combination of transformations (as a tuple) that 
        produces the most class separability
    '''
    class1_imgs = [ensure_rgb(img) for img in class1_imgs]
    class2_imgs = [ensure_rgb(img) for img in class2_imgs]
    # normalize all images to [0, 255] range
    class1_imgs = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) for img in class1_imgs]
    class2_imgs = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) for img in class2_imgs]

    combos = powerset_without_emptyset(transformations)
    
    score_dict = {}
    for combo in tqdm(combos, total=len(combos)):
        
        transformed1 = apply_transformations(class1_imgs, combo)
        transformed2 = apply_transformations(class2_imgs, combo)
        
        inter_score = nan_euclidean_distances(transformed1, transformed2).mean()
        intra_score1 = nan_euclidean_distances(transformed1, np.flip(transformed1, axis=0)).mean()
        intra_score2 = nan_euclidean_distances(transformed2, np.flip(transformed2, axis=0)).mean()
        score_dict[combo] = {"inter-score": inter_score, "intra-score class 1": intra_score1, "intra-score class 2": intra_score2}

    return score_dict


def contrastive_loss(pos_pairs, neg_pairs, margin=.5):
    """
    pos_pairs: tensor of shape (N, D) for positive pairs (same class)
    neg_pairs: tensor of shape (N, D) for negative pairs (different class)
    """
    # euclidean distance 
    pos_dist = torch.norm(pos_pairs[:, 0] - pos_pairs[:, 1], dim=1)
    neg_dist = torch.norm(neg_pairs[:, 0] - neg_pairs[:, 1], dim=1)

    pos_loss = torch.mean(pos_dist ** 2)
    # make the embeddings from different classes at least `margin` apart 
    neg_loss = torch.mean(torch.clamp(margin - neg_dist, min=0) ** 2)
    
    return pos_loss + neg_loss