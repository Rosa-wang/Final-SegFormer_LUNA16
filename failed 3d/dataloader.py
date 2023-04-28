from configs import VAL_PCT, OUTPUT_PATH, BATCH_SZIE
import random
import itertools
import pandas as pd
from dataset import LunaDataSet
from torch.utils.data import DataLoader

def build_dataloader():
    meta = pd.read_csv(f'{OUTPUT_PATH}/augmented_meta.csv', index_col=0).sample(frac=1).reset_index(drop=True)
    # meta = pd.read_csv(f'{OUTPUT_PATH}/preprocessed_meta.csv', index_col=0).sample(frac=1).reset_index(drop=True)
    meta_group_by_series = meta.groupby(['seriesuid']).indices
    list_of_groups = [{i: list(meta_group_by_series[i])} for i in meta_group_by_series.keys()]
    random.Random(0).shuffle(list_of_groups)
    
    
    fold_size = len(list_of_groups) // 10
    fold_indices = [list_of_groups[i:i+fold_size] for i in range(0, len(list_of_groups), fold_size)]

    train_loaders = []
    val_loaders = []

    for i in range(len(fold_indices)):
        val_groups = fold_indices[i]
        val_indices = list(itertools.chain(*[list(i.values())[0] for i in val_groups]))
        train_groups = [g for j, g in enumerate(fold_indices) if j != i]
        train_indices = list(itertools.chain(*[list(i.values())[0] for g in train_groups for i in g]))

        ltd = LunaDataSet(train_indices, meta)
        lvd = LunaDataSet(val_indices, meta)
        train_loader = DataLoader(ltd, batch_size=BATCH_SZIE, shuffle=False)
        val_loader = DataLoader(lvd, batch_size=BATCH_SZIE, shuffle=False)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders

    # val_split = int(VAL_PCT * len(list_of_groups))
    # val_indices = list(itertools.chain(*[list(i.values())[0] for i in list_of_groups[:val_split]]))
    # train_indices = list(itertools.chain(*[list(i.values())[0] for i in list_of_groups[val_split:]]))
    # ltd = LunaDataSet(train_indices, meta)
    # lvd = LunaDataSet(val_indices, meta)
    # train_loader = DataLoader(ltd, batch_size=BATCH_SZIE, shuffle=True)
    # val_loader = DataLoader(lvd, batch_size=BATCH_SZIE, shuffle=False)
    # return train_loader, val_loader