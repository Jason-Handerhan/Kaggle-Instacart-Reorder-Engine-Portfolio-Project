import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'utilities'))

from utility_functions import find_near_duplicate_names


def test_returns_similar_pairs_once_highest_score_first():
    names = ['Organic Bananas', 'Organic Banana', 'Whole Milk', 'Organic Bananas ', 'Dish Soap']

    result = find_near_duplicate_names(names, threshold=90)

    assert [(a, b) for a, b, _ in result] == [
        ('Organic Bananas', 'Organic Bananas '),
        ('Organic Bananas', 'Organic Banana'),
        ('Organic Banana', 'Organic Bananas '),
    ]
    assert [score for _, _, score in result] == sorted([s for _, _, s in result], reverse=True)


def test_threshold_filters_pairs():
    assert find_near_duplicate_names(['Whole Milk', 'Dish Soap'], threshold=90) == []
    assert len(find_near_duplicate_names(['Whole Milk', 'Dish Soap'], threshold=0)) == 1
