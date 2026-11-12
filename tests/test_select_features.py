import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utilities.utility_functions import select_features


def build_df():
    columns = [
        "user_prod_avg_days_between_purchases", "user_product_consecutive_order_skips",
        "user_product_order_penetration_ratio", "user_product_reorder_concentration",
        "label_user_product_days_since_last_purchased", "user_product_last3order_streak",
        "user_product_cumulative_purchases", "user_product_purchase_concentration",
        "user_product_nearing_reorder_ratio", "user_reorder_ratio",
        "user_product_cart_position_last_order", "user_product_avg_cart_position",
        "user_distinct_product_count", "user_deptartment_share", "user_aisle_share",
        "user_aisle_pacing_ratio", "user_id", "anchor_order_number", "label_reordered",
    ]
    return pd.DataFrame([[1] * len(columns)], columns=columns)


def test_aisle_pacing_ratio_is_opt_in():
    df = build_df()

    _, default_features, _ = select_features(df)
    ranker_df, ranker_features, _ = select_features(df, include_aisle_pacing=True)

    assert "user_aisle_pacing_ratio" not in default_features
    assert ranker_features == default_features + ["user_aisle_pacing_ratio"]
    assert "user_aisle_pacing_ratio" in ranker_df.columns
