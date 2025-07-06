from pathlib import Path
from torchvision.io import decode_image
from torch.utils.data import Dataset 
import os
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

# create a Dataset class to retrieve the data
class SQLDataset_Humanitarian(Dataset):
    def __init__(self, conn, label_col, img_col='image_path', data_dir=Path(os.path.expanduser('~'), 'CrisisMMD_v2.0','CrisisMMD_v2.0'), 
                 transform=None, target_transform=None, is_train=False, is_test=False, is_val=False, table_name='Images'):
        '''
        Parameters: 
            conn - a mysql.connector object that will be used to retrieve a cursor 
            label_col - a name of type string that matches the column name in the sql database that labels the image data
            img_col - a name of type string that matches the column name in the sql database that contains the image path
            data_dir - the path that contains the folder containing the image paths in the sql database 
            transform - pytorch image transformations that transform the data
            target_transform - does nothing as of now, so do not specify this
            is_train | is_val | is_test - choose none or one of these; if none chosen, all data is used 
            table_name - a string that matches the table name; default: "Images"
        
        Notes:
            is_train uses 90% of the data
            is_val and is_test each use 5% of the data 
        '''
        assert(not ((is_train and is_test) or (is_train and is_val) or (is_val and is_test)), 'a dataset can only be one of either train, test, or val')

        self.conn = conn
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = data_dir
        self.is_train = is_train
        self.is_val = is_val
        self.is_test = is_test
        self.table_name = table_name

        cursor = self.conn.cursor()

        # we need a list of available indices 
        # what idxs are available to use for this database? - depends on the dataset type
        if not (self.is_train or self.is_val or self.is_test): # if no dataset type specified
            query = 'SELECT COUNT(image_id) FROM ' + table_name
            cursor.execute(query)
            count = cursor.fetchone()[0]
            self.possible_sql_idxs = range(count)
        else:
            
            query = 'SELECT COUNT(image_id) FROM ' + table_name
            cursor.execute(query)
            count = cursor.fetchone()[0]
            

            # below, we use +1 because SQL indexing starts at 1
            if is_train: 
                self.possible_sql_idxs = [i+1 for i in range(count) if (((i+1) % 20) < 18)]
            elif is_val:
                self.possible_sql_idxs = [i+1 for i in range(count) if (((i+1) % 20) == 18)]
            else: # must be test
                self.possible_sql_idxs = [i+1 for i in range(count) if (((i+1) % 20) > 18)]
        cursor.close()

    def __len__(self):
        '''
        Returns the number of images in the database 
        '''
    

        return len(self.possible_sql_idxs)
    
    # TODO: change this so that it uses the humanitarian classes you are using
    
    def __getitem__(self, idx):
        '''
        Description: 
            Retrieves a tuple of (torch.tensor, string) where the first object is a 3D tensor of image data and the string is the label
        '''
        # retrieve an image from the sql database
        return SyntaxError('You need to change this to use humanitarian classes and label column')

        cursor = self.conn.cursor()

        try:
                query = f'SELECT {self.img_col}, {self.label_col} FROM {self.table_name} WHERE idx={self.possible_sql_idxs[idx]}' 
                cursor.execute(query)
                
                # read in image
                img_path, label = cursor.fetchone()
                img_path = Path(self.data_dir, img_path)
                image = ensure_rgb(np.array(Image.open(img_path))) # channels-last bc cv2
        

                
                if label == 'informative':
                    label = torch.tensor(1)
                else:
                    label = torch.tensor(0)
                # print(f'image shape before transform: {image.shape}')
                # apply transforms on image 
                if self.transform:
                    image = self.transform(image)
                if self.target_transform:
                    label = self.target_transform(label)
        finally:
            cursor.close()

        return {'image': image, 'label': label}
    



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
            assert(len(img.expand_dims(axis=0).shape)==3, f"wrong shaped image: {img.shape}")
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).expand_dims(axis=0)
        elif img.ndim == 3 and img.shape[2] == 3:
            assert(len(img.shape)==3, f"wrong shaped image: {img.shape}")
            return img  # Already RGB
        else:
            raise ValueError("Unsupported image shape for ensure_rgb: {}".format(img.shape))
    else:
        raise TypeError("Input must be a numpy ndarray.")
    

def apply_transformations(images, combo):
    features = []
    for t in combo:
        if t is cv2.HuMoments:
            imgs_grey = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img for img in images]
            feats = [cv2.normalize(cv2.HuMoments(cv2.moments(im)).flatten(), None, 0, 255, cv2.NORM_MINMAX) for im in imgs_grey]
        elif t is skimage.feature.graycomatrix:
            imgs_grey = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img for img in images]
            dists, angles = [1], [0]
            feats = []
            for im in imgs_grey:
                glcm = skimage.feature.graycomatrix(im, distances=dists, angles=angles, symmetric=True, normed=True)
                diss = graycoprops(glcm, 'dissimilarity')
                contrast = graycoprops(glcm, 'contrast')
                cat = np.concatenate([diss, contrast], axis=1)
                norm = cv2.normalize(cat, None, 0, 255, cv2.NORM_MINMAX).flatten()
                feats.append(norm)
        elif t is cv2.calcHist:
            feats = [cv2.normalize(
                        cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256]*3).flatten(),
                        None, 0, 255, cv2.NORM_MINMAX).flatten() for img in images]
        elif t is skimage.feature.local_binary_pattern:
            imgs_grey = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img for img in images]
            feats = []
            for im in imgs_grey:
                lbp = skimage.feature.local_binary_pattern(im, P=8, R=1)
                hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
                feats.append(hist.flatten())
        else:
            print(t)
            raise ValueError(f'Unsupported transformation: {t}')
        
        features.append(np.stack(feats))

        for i, feature in enumerate(features): 
            if len(feature.shape) > 1:
                features[i] = feature.flatten()
                
    return np.concatenate(features, axis=0)

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
    class1_imgs = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) for img in class1_imgs]
    class2_imgs = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) for img in class2_imgs]

    combos = powerset_without_emptyset(transformations)
    best_combo = None
    
    score_dict = {}
    for combo in tqdm(combos, total=len(combos)):
        
        transformed1 = np.expand_dims(apply_transformations(class1_imgs, combo), axis=1)
        transformed2 = np.expand_dims(apply_transformations(class2_imgs, combo), axis=1)
        
        inter_score = nan_euclidean_distances(transformed1, transformed2).mean()
        intra_score1 = nan_euclidean_distances(transformed1, np.flip(transformed1, axis=0))
        intra_score2 = nan_euclidean_distances(transformed2, np.flip(transformed2, axis=0))
        score_dict[combo] = {"inter-score": inter_score, "intra-score class 1": intra_score1, "intra-score class 2": intra_score2}

    return score_dict