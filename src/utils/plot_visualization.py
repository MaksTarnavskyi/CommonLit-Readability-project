import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')

def plot_histogram(df: pd.DataFrame, target_column: str, x_label: str = '', y_label:str = '', title = ''):
    p = df[target_column].hist()
    p.set_xlabel(x_label)
    p.set_ylabel(y_label)
    p.set_title(title)
    return p
