import sys
sys.path.insert(0,'../functions')

from functions import save_sentinel_features_wrapper

import pandas as pd

from multiprocessing import Manager, Pool
from tqdm import tqdm

def main():
    metadata = pd.read_csv('../data/metadata.csv')

    manager = Manager()
    features = manager.list()
    failed_points = manager.list()

    head = len(metadata)

    total_rows = len(metadata.head(head))

    args = [(r, failed_points, features) for _, r in metadata.head(head).iterrows()]

    # Crear un Pool de procesos
    with Pool(processes=32) as pool:
        # Utilizar tqdm para la barra de progreso
        with tqdm(total=total_rows) as pbar:
            # Mapear la función sobre los argumentos
            for _ in pool.imap_unordered(save_sentinel_features_wrapper, args):
                pbar.update(1)

    features_df = pd.DataFrame.from_records(features)
    failed_points_df = pd.DataFrame.from_records(failed_points)
    features_df.to_csv('../data/downloaded/sentinel/features.csv')
    failed_points_df.to_csv('../data/downloaded/failed/failed_points_sentinel.csv')

main()
