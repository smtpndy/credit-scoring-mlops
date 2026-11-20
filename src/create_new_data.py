"""Task A (part 1): create a small incoming data batch.

Real life: new labelled customers arrive after some months.
Demo: we take 8 REAL rows from the old TEST split (the model never trained on
them) and save them as datasets/new/new_batch.csv.
"""
import pandas as pd

from src.common import DATA_PATH, ID_COLUMN, NEW_DATA_PATH, TARGET, load_original_split


def main():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = load_original_split()

    # rows of the old test split (X_test.index = row number in the csv)
    test_rows = df.loc[X_test.index]

    # pick 3 bad-credit and 5 good-credit customers (fixed seed = same every run)
    bad = test_rows[test_rows[TARGET] == 0].sample(3, random_state=1)
    good = test_rows[test_rows[TARGET] == 1].sample(5, random_state=1)
    new_batch = pd.concat([bad, good]).sort_index()

    # add an id column so we can check for duplicate ids later
    new_batch.insert(0, ID_COLUMN, new_batch.index)

    NEW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_batch.to_csv(NEW_DATA_PATH, index=False)

    print("New batch created:", NEW_DATA_PATH)
    print("Rows:", len(new_batch))


if __name__ == "__main__":
    main()
