from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
import numpy as np
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import math
from configs import BLOCK_SIZE, TARGET_SHAPE, ANCHOR_SIZES, OUTPUT_PATH, PADDING_FOR_LOCALIZATION, COORDS_CUBE_SIZE


class LunaDataSet(Dataset):
    def __init__(self, indices: list, meta_dataframe: pd.DataFrame):
        self.indices = indices
        self.meta_dataframe = meta_dataframe

    def __getitem__(self, idx, split=None):
        meta = self.meta_dataframe.iloc[self.indices[idx]]
        centers = literal_eval(meta['centers'])
        radii = literal_eval(meta['radii'])
        # lungs_bounding_box = literal_eval(meta['lungs_bounding_box'])
        clazz = int(meta['class'])
        sub_dir = 'positives' if clazz == 1 else 'negatives'
        file_path = f'''{OUTPUT_PATH}/augmented/{sub_dir}/{meta['seriesuid']}_{meta['sub_index']}.npy'''
        # file_path = f'''{OUTPUT_PATH}/preprocessed/{sub_dir}/{meta['seriesuid']}.npy'''
        patch = np.load(file_path)
        target = np.zeros(TARGET_SHAPE)
        # if clazz == 1:
        #     for c in range(len(centers)):
        #         place = []
        #         point = []
        #         windows = []
        #         for ax in range(len(patch.shape)):
        #             window = int(BLOCK_SIZE / TARGET_SHAPE[ax])
        #             windows.append(window)
        #             place.append(centers[c][ax] // window)
        #             point.append(centers[c][ax] % window)
        #         import pdb; pdb.set_trace()
        #         if radii[c] <= ANCHOR_SIZES[0] / 2:
        #             place.append(0)
        #         elif radii[c] <= ANCHOR_SIZES[1] / 2:
        #             place.append(1)
        #         else:
        #             place.append(2)
        #         vector = [1]
        #         for p in range(len(point)):
        #             vector.append(point[p] / windows[p] - 1)
        #         vector.append(radii[c])
        #         import pdb; pdb.set_trace()
        #         target[tuple(place)] = vector
        # else:
        #     for c in range(len(centers)):
        #         point = []
        #         for ax in range(len(patch.shape)):
        #             window = int(BLOCK_SIZE / TARGET_SHAPE[ax])
        #             point.append(centers[c][ax] % window)       
        if clazz == 1:
            for c in range(len(centers)):
                place = []
                windows = []
                for ax in range(len(patch.shape)):
                    window = int(BLOCK_SIZE / TARGET_SHAPE[ax])
                    windows.append(window)
                    place.append(centers[c][ax] // window)
                x, y, z = np.indices(patch.shape)
                # x, y, z = np.indices(target.shape)
                indices = np.stack([x, y, z], axis=-1)
                # place =(place[0], place[1], place[2])       
                dist_vec = indices - np.array(place)
                dist = np.linalg.norm(dist_vec, ord=2, axis=-1)
                mask = dist <= radii[c]
                target[mask] = 1
        else:
            for c in range(len(centers)):
                point = []
                for ax in range(len(patch.shape)):
                    window = int(BLOCK_SIZE / TARGET_SHAPE[ax])
                    point.append(centers[c][ax] % window)
        out_patch = patch[np.newaxis, ]      
        return out_patch, target

    def __len__(self):
        return len(self.indices)
