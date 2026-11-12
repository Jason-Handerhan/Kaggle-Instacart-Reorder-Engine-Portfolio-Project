from src.utilities.utility_functions import find_near_duplicate_names


def test_find_near_duplicate_names():
    names = ["Organic Banana", "Organic Bananas", "Whole Milk", "Organic Banan"]
    result = find_near_duplicate_names(names, threshold=90)

    assert [(a, b) for a, b, _ in result] == [
        ("Organic Banana", "Organic Bananas"),
        ("Organic Banana", "Organic Banan"),
        ("Organic Bananas", "Organic Banan"),
    ]
    assert result[0][2] >= result[1][2] >= result[2][2] >= 90
