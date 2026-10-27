import pandas as pd
import pytest
from kestrel.schema import choose_data_files

def test_unknown_files_cannot_silently_guess(tmp_path):
    pd.DataFrame({'ID':['a'], 'team':['A']}).to_csv(tmp_path/'sample_submission.csv',index=False)
    with pytest.raises(ValueError,match='Cannot safely identify'):
        choose_data_files(tmp_path,{})
