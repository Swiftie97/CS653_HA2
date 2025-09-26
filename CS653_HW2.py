import pandas as pd
from pathlib import Path

def read_files():
    pd.set_option("mode.copy_on_write", True)
    cwd = Path().cwd()
    wine_quality_red_filename = Path('winequality-red.csv')
    wine_quality_white_filename = Path('winequality-white.csv')
    
    if not wine_quality_red_filename.exists():
        raise FileNotFoundError(f"Dataset file not found: {wine_quality_red_filename}")
    if not wine_quality_white_filename.exists():
        raise FileNotFoundError(f"Dataset file not found: {wine_quality_white_filename}")

    wine_red = pd.read_csv(wine_quality_red_filename)
    wine_white = pd.read_csv(wine_quality_white_filename)

    return (wine_red, wine_white)