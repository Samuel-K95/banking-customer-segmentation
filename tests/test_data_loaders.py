import pytest
import pandas as pd
from src.data.indian_bank_dataset_loader import load_indian_bank_raw_data

def test_indian_bank_loader_handles_missing_file():
    with pytest.raises(FileNotFoundError):
        load_indian_bank_raw_data("non_existent_file.csv")

# Note: We avoid testing the actual data loading without mocking 
# to prevent tests from taking too long or failing on missing huge CSVs.
