import pandas as pd
import glob
import os

# Define the directory containing the CSV files
data_dir = 'dataset/stock_csv/'  # Ensure this path is correct
output_file = 'summary_statistics_2015_2017.csv'

# Define the date range
start_date = '2015-01-01'
end_date = '2017-12-31'

# Define the columns to analyze
columns_to_analyze = [
    'Daily Return',
    'MA-10',
    'EMA',
    'Volatility-10',
    'RSI',
    'Volume MA-20',
    'High-Low',
    'Close-Open',
    'ROC-10',
    'Log Returns'
]

# Initialize a list to store summary data
summary_data = []

# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
print(f"Found {len(csv_files)} CSV files in '{data_dir}'.")

for file_path in csv_files:
    try:
        # Extract ticker from filename
        # Example filename: 'AAPL.us.csv'
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]  # 'AAPL.us'
        
        # Split at '.us' to get the ticker
        if '.us' in name_without_ext:
            ticker = name_without_ext.split('.us')[0]
        else:
            # If '.us' is not present, use the entire name_without_ext as ticker
            ticker = name_without_ext
        
        print(f"\nProcessing ticker: {ticker} from file: {base_name}")
        
        # Read the CSV file
        df = pd.read_csv(file_path, parse_dates=['Date'])
        total_rows = len(df)
        print(f"Total rows in {ticker}: {total_rows}")
        
        # Filter rows within the date range
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df_filtered = df.loc[mask]
        filtered_rows = len(df_filtered)
        print(f"Rows after filtering by date ({start_date} to {end_date}): {filtered_rows}")
        
        if df_filtered.empty:
            print(f"No data for {ticker} in the specified date range. Skipping.")
            continue
        
        # Initialize a dictionary to store statistics for the current ticker
        ticker_stats = {'Ticker': ticker}
        
        for col in columns_to_analyze:
            if col not in df_filtered.columns:
                print(f"Column '{col}' not found in {ticker}. Assigning None for its statistics.")
                ticker_stats[f'{col}_25th'] = None
                ticker_stats[f'{col}_50th'] = None
                ticker_stats[f'{col}_75th'] = None
                continue
            
            # Drop NaN values for accurate statistics
            data = df_filtered[col].dropna()
            
            if data.empty:
                print(f"All values are NaN for column '{col}' in {ticker}. Assigning None for its statistics.")
                ticker_stats[f'{col}_25th'] = None
                ticker_stats[f'{col}_50th'] = None
                ticker_stats[f'{col}_75th'] = None
                continue
            
            # Calculate statistics
            q25 = data.quantile(0.25)
            q50 = data.quantile(0.50)
            q75 = data.quantile(0.75)
            
            ticker_stats[f'{col}_25th'] = q25
            ticker_stats[f'{col}_50th'] = q50
            ticker_stats[f'{col}_75th'] = q75
            
            print(f"{ticker} - {col}: 25th={q25}, 50th={q50}, 75th={q75}")
        
        # Append the ticker's statistics to the summary list
        summary_data.append(ticker_stats)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Check if summary_data has any entries
total_processed = len(summary_data)
print(f"\nTotal tickers processed: {total_processed}")

if total_processed == 0:
    print("No data was processed. Please check the date range and data availability.")
else:
    # Convert the summary data to a DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Define the desired column order
    ordered_columns = ['Ticker']
    for col in columns_to_analyze:
        ordered_columns.extend([
            f"{col}_25th",
            f"{col}_50th",
            f"{col}_75th"
        ])
    
    # Reorder the DataFrame columns accordingly
    # Handle cases where some columns might not be present by intersecting with existing columns
    ordered_columns = [col for col in ordered_columns if col in summary_df.columns]
    summary_df = summary_df[ordered_columns]
    
    # Save the summary to a CSV file
    summary_df.to_csv(output_file, index=False)
    
    print(f"Summary statistics saved to '{output_file}'.")
