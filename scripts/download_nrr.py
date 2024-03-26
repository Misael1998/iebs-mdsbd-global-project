import sys
sys.path.insert(0,'../functions')

from functions import save_features_wrapper

import pandas as pd

from multiprocessing import Manager, Pool
from tqdm import tqdm

def main():
    metadata = pd.read_csv('../data/metadata.csv')
    failed_points = pd.read_csv('../data/downloaded/failed/sentinel.csv')

    valid_points = pd.merge(metadata, failed_points['uid'], on='uid', how='left', indicator=True)
    valid_points = valid_points[valid_points['_merge'] == 'left_only']

    # Eliminar la columna auxiliar _merge
    valid_points = valid_points.drop('_merge', axis=1)

    dates = valid_points.date
    dates = dates.drop_duplicates()

    # Utilizamos una lista compartida para almacenar los puntos válidos
    manager = Manager()
    features = manager.list()
    failed_dates = manager.list()

    head = len(dates)

    # Obtener el número total de filas
    total_rows = len(dates.head(head))

    # Crear un iterable de argumentos para el método map
    args = [(date, valid_points, features, failed_dates) for date in dates.head(head)]

    # Crear un Pool de procesos
    with Pool(processes=16) as pool:
        # Utilizar tqdm para la barra de progreso
        with tqdm(total=total_rows) as pbar:
            # Mapear la función sobre los argumentos
            for _ in pool.imap_unordered(save_features_wrapper, args):
                pbar.update(1)

    features_df = pd.DataFrame.from_records(features)
    failed_dates_df = pd.DataFrame.from_records(failed_dates)
    features_df.to_csv('../data/downloaded/nrr/nrr_features.csv')
    failed_dates_df.to_csv('../data/downloaded/failed/failed_dates_nrr.csv')

main()
