import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utilities.utility_functions import select_features


def test_default_feature_list_excludes_aisle_pacing_ratio():
    _, feature_list, _ = select_features(MagicMock())

    assert "user_aisle_pacing_ratio" not in feature_list


def test_ranker_feature_list_adds_only_aisle_pacing_ratio():
    _, default_features, _ = select_features(MagicMock())
    df = MagicMock()
    _, ranker_features, _ = select_features(df, include_aisle_pacing=True)

    assert ranker_features == default_features + ["user_aisle_pacing_ratio"]
    # the ratio is also kept as a column of the returned frame
    assert "user_aisle_pacing_ratio" in df.__getitem__.call_args.args[0]
