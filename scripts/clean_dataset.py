import pandas as pd
import os

def main():
    print("Select the dataset")
    print("1. HI_Small")
    print("2. HI_Large")

    choice = input("Enter the number of your choice: ").strip()

    if choice == '1':
        dataset_folder = 'HI_Small'
        cutoff_date = '2022-09-11'
    elif choice == '2':
        dataset_folder = 'HI_Large'
        cutoff_date = '2022-11-06'
    else:
        print("Invalid selection. Exiting.")
        return

    if os.path.exists('./data'):
        base_path = './data'
    elif os.path.exists('../data'):
        base_path = '../data'
    else:
        print("Could not find data directory.")
        return

    input_path = os.path.join(base_path, dataset_folder, '2_filtered_laundering_transactions.parquet')
    output_path = os.path.join(base_path, dataset_folder, 'cleaned_laundering_transactions.parquet')

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)

    print(f"Filtering transactions <= {cutoff_date}...")
    df = df[df['timestamp'] <= cutoff_date]

    print(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)

    print(f"Min timestamp: {df['timestamp'].min()}")
    print(f"Max timestamp: {df['timestamp'].max()}")

if __name__ == "__main__":
    main()