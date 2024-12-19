import pandas as pd
import argparse

def load_csv(file_path):
    """Load a CSV file and handle potential errors."""
    try:
        return pd.read_csv(file_path, sep=';')
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"File '{file_path}' is empty.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print(f"Error parsing file '{file_path}'.")
        return pd.DataFrame()

def search_in_data(df, search_col, search_value):
    """Search for rows in the DataFrame based on a specified column and value."""
    if search_col not in df.columns:
        print(f"Column '{search_col}' not found.")
        return pd.DataFrame()

    df[search_col] = df[search_col].astype(str)  # Ensure values are strings
    results = df[df[search_col].str.contains(search_value, case=False, na=False)]
    return results

def print_results(results, output_cols):
    """Print the results of the search."""
    available_cols = [col for col in output_cols if col in results.columns]
    
    if not results.empty and available_cols:
        print(results[available_cols])
    else:
        print("No matches found or output columns are not available.")



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Search in a CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')
    parser.add_argument('search_col', type=str, help='Column to search')
    parser.add_argument('search_value', type=str, help='Value to search for')
    parser.add_argument('--output_cols', type=str, nargs='+', help='Columns to output', default=['ACR', 'Bi-Rads'])
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    df = load_csv(args.file_path)
    
    if df.empty:
        exit()  # If DataFrame is empty, exit the program
    
    found_rows = search_in_data(df, args.search_col, args.search_value)
    print_results(found_rows, args.output_cols)

# example:
#     python parser.py "INbreast Release 1.0/INbreast.csv" "File Name" "22678646" --output_cols ACR Bi-Rads