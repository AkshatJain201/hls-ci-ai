import pandas as pd
import numpy as np
import os

def split_excel_file(input_file, output_folder, output_prefix):
    # Read the original Excel file
    df = pd.read_excel(input_file)

    # Calculate the size of each chunk
    chunk_size = len(df) // 10

    # Split the dataframe into 5 parts
    for i in range(10):
        # Determine the start and end index for each chunk
        start_index = i * chunk_size
        if i == 9:  # For the last chunk, take the remaining rows
            end_index = len(df)
        else:
            end_index = (i + 1) * chunk_size
        
        # Extract the chunk
        chunk_df = df.iloc[start_index:end_index]

        # Create a new file name
        output_file = os.path.join(output_folder, f"{output_prefix}_{i+1}.xlsx")

        # Save the chunk to a new Excel file
        chunk_df.to_excel(output_file, index=False)
        print(f"Saved: {output_file}")

input_file = './data/06-12-2024.xlsx'
output_folder = './data'
split_excel_file(input_file, output_folder, 'output')

