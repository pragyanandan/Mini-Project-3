
import pandas as pd
import numpy as np

def get_columns_with_nulls(df: pd.DataFrame) -> list:
    """
    Returns a list of dictionaries, where each dictionary contains a column name
    with null values and the count of null values in that column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    list: A list of dictionaries with column names and their respective null value counts.
    """
    null_columns = df.isnull().sum()
    result = [{col: null_columns[col]} for col in null_columns.index if null_columns[col] > 0]
    
    return result


def replace_nulls_with_random(df, column_name, replacement_values):
    """
    Replace null (NaN) values in a specified column of a DataFrame with random values from a given array.

    :param df: pandas DataFrame
    :param column_name: The name of the column to perform replacement on
    :param replacement_values: numpy array of values to randomly pick from
    :return: pandas DataFrame with null values replaced
    """
    # Ensure the replacement_values is a numpy array
    replacement_values = np.array(replacement_values)
    
    # Find the indices where the column has NaN values
    null_indices = df[df[column_name].isnull()].index
    
    # Replace NaN values with random choices from replacement_values
    df.loc[null_indices, column_name] = np.random.choice(replacement_values, size=len(null_indices))
    
    return df
