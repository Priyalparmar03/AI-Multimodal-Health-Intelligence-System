import pandas as pd

def preprocess_dataset():
    df = pd.read_csv("data/dataset.csv")

    # Fill missing values
    df.fillna("None", inplace=True)

    # Collect all unique symptoms
    symptoms = set()
    for col in df.columns[1:]:
        symptoms.update(df[col].unique())

    symptoms.discard("None")
    symptoms = list(symptoms)

    # Create new dataset
    rows = []
    for _, row in df.iterrows():
        row_dict = dict.fromkeys(symptoms, 0)

        for col in df.columns[1:]:
            if row[col] != "None":
                row_dict[row[col]] = 1

        row_dict["disease"] = row["Disease"]
        rows.append(row_dict)

    new_df = pd.DataFrame(rows)
    new_df.to_csv("data/processed_dataset.csv", index=False)

    print(" Dataset converted to numeric format!")