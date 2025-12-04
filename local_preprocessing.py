import numpy as np
import pandas as pd

def local_preprocess(df : pd.DataFrame,_scaler = None):
    """
    This function processes a DataFrame - fills np.nan with median
    and returns a DataFrame with numerical values, it also converts type to numerical
    :param df: A dataset for prediction
    :param _scaler: A MinMaxScaler for scale (if None - no scaling is applied)
    :return: A preprocessed DataSet with numeric values
    """
    required_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    df_new = df.copy()

    # Функция для преобразования типов данных (по аналогии с ДЗ-1)
    # Признаюсь честно - DeepSeek частично помог затюнить эту функцию
    def str_to_float(value):
        if isinstance(value, (int, float, np.number)):
            return float(value)

        if isinstance(value, str):
            value = value.strip()
            if ' ' in value:
                parts = value.split()
                if parts:
                    try:
                        return float(parts[0])
                    except (ValueError, TypeError):
                        pass
            try:
                return float(value)
            except (ValueError, TypeError):
                return np.nan

        return np.nan

    if not all(col in df_new.columns for col in required_columns):
        raise Exception("Can't find required columns in this Dataframe")

    for column in required_columns:
        if df_new[column].dtype.name not in ['int64', 'int32', 'float64', 'float32']:
            df_new[column] = df_new[column].apply(str_to_float)
        med = df_new[column].median(skipna=True)
        df_new[column].fillna(med,inplace=True)

    if _scaler is not None:
        df_new = _scaler.transform(df_new[required_columns])
        return df_new

    return df_new[required_columns]







