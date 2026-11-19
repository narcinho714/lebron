import pandas as pd
from foulup3.simple import table

def test_table_pools_and_diffs():
    df = pd.DataFrame({"t_entry": [3, 3, 3, 3], "foul": [1, 1, 0, 0], "leader_win": [1, 1, 1, 0]})
    t = table(df).set_index("sec")
    assert t.loc["All", "n_foul"] == 2 and t.loc["All", "n_defend"] == 2
    assert abs(t.loc["All", "diff"] - 50) < 1e-9
