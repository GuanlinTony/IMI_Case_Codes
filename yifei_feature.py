"""
AML Feature Engineering - Selected Features
============================================

Selected features:
1. num_channels_used
2. late_night_ratio
3. late_night_txn_count
4. weekend_txn_count
5. activity_span_days
6. unique_provinces
7. unique_countries
8. unique_cities
9. geo_diversity_score
10. near_threshold_count
11. avg_txn_per_active_day
12. unique_merchant_categories
13. txn_to_income_ratio
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


def load_all_data(data_path: str = './') -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the competition data.
    """
    data = {}

    # Transaction files
    transaction_files = ['abm', 'card', 'cheque', 'eft', 'emt', 'westernunion', 'wire']
    for file in transaction_files:
        try:
            data[file] = pd.read_parquet(f'{data_path}/{file}.parquet')
            print(f"Loaded {file}.parquet: {len(data[file]):,} rows")
        except FileNotFoundError:
            print(f"Warning: {file}.parquet not found")

    # KYC files
    kyc_files = ['kyc_individual', 'kyc_smallbusiness', 'kyc_occupation_codes', 'kyc_industry_codes']
    for file in kyc_files:
        try:
            data[file] = pd.read_parquet(f'{data_path}/{file}.parquet')
            print(f"Loaded {file}.parquet: {len(data[file]):,} rows")
        except FileNotFoundError:
            print(f"Warning: {file}.parquet not found")

    # Labels
    try:
        data['labels'] = pd.read_parquet(f'{data_path}/labels.parquet')
        print(f"Loaded labels.parquet: {len(data['labels']):,} rows")
    except FileNotFoundError:
        print("Warning: labels.parquet not found")

    return data


def preprocess_transactions(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """
    Preprocess transaction data with standardized columns and datetime parsing.
    """
    df = df.copy()
    df['channel'] = channel
    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
    df['date'] = df['transaction_datetime'].dt.date
    df['hour'] = df['transaction_datetime'].dt.hour
    df['day_of_week'] = df['transaction_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    return df


def combine_transactions(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all transaction channels into a single dataframe.
    """
    channels = ['abm', 'card', 'cheque', 'eft', 'emt', 'westernunion', 'wire']

    all_transactions = []
    for channel in channels:
        if channel in data:
            df = preprocess_transactions(data[channel], channel)
            all_transactions.append(df)

    combined = pd.concat(all_transactions, ignore_index=True)
    combined = combined.sort_values(['customer_id', 'transaction_datetime'])

    return combined



def compute_temporal_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Features: late_night_ratio, late_night_txn_count, weekend_txn_count, activity_span_days
    """
    features = transactions.groupby('customer_id').agg({
        'hour': lambda x: (x < 6).sum(),  # late_night_txn_count (midnight-6am)
        'is_weekend': 'sum',  # weekend_txn_count
        'transaction_datetime': ['min', 'max'],
        'transaction_id': 'count'  # total transactions for ratio
    })

    features.columns = ['late_night_txn_count', 'weekend_txn_count', 'first_txn', 'last_txn', 'total_txn']

    # activity_span_days
    features['activity_span_days'] = (
                                             features['last_txn'] - features['first_txn']
                                     ).dt.days + 1

    # late_night_ratio
    features['late_night_ratio'] = features['late_night_txn_count'] / features['total_txn']

    # Select final columns
    features = features[['late_night_txn_count', 'late_night_ratio', 'weekend_txn_count', 'activity_span_days']]

    return features.reset_index()





def compute_unique_merchant_categories(card_transactions: pd.DataFrame,
                                       all_customer_ids: pd.Series) -> pd.DataFrame:
    """
    Feature: unique_merchant_categories
    Number of unique merchant categories from card transactions.
    """
    all_customers = pd.DataFrame({'customer_id': all_customer_ids})

    if len(card_transactions) == 0 or 'merchant_category' not in card_transactions.columns:
        all_customers['unique_merchant_categories'] = 0
        return all_customers

    features = card_transactions.groupby('customer_id')['merchant_category'].nunique().reset_index()
    features.columns = ['customer_id', 'unique_merchant_categories']

    all_customers = all_customers.merge(features, on='customer_id', how='left')
    all_customers['unique_merchant_categories'] = all_customers['unique_merchant_categories'].fillna(0)

    return all_customers


def run_feature_engineering(data_path: str = './', output_path: str = None) -> pd.DataFrame:
    """
    Main function to run feature engineering for selected features only.

    Args:
        data_path: Path to directory containing the parquet files
        output_path: Path for output CSV file. If None, saves to data_path/aml_features_selected.csv

    Returns:
        DataFrame with computed features
    """
    print("=" * 60)
    print("AML Feature Engineering - Selected Features")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    data = load_all_data(data_path)

    # Check if we have any transaction data
    transaction_channels = ['abm', 'card', 'cheque', 'eft', 'emt', 'westernunion', 'wire']
    available_channels = [ch for ch in transaction_channels if ch in data]

    if not available_channels:
        raise ValueError(
            f"No transaction data found in {data_path}!\n"
            f"Expected parquet files: {', '.join([f'{ch}.parquet' for ch in transaction_channels])}"
        )

    print(f"\n   Available channels: {', '.join(available_channels)}")

    # Combine transactions
    print("\n2. Combining transactions...")
    transactions = combine_transactions(data)
    print(f"   Total transactions: {len(transactions):,}")

    # Initialize feature dataframe
    all_customers = pd.DataFrame({
        'customer_id': transactions['customer_id'].unique()
    })
    print(f"   Total customers: {len(all_customers):,}")

    # Compute selected features
    print("\n3. Computing selected features...")

    print("   - late_night_ratio, late_night_txn_count, weekend_txn_count, activity_span_days")
    temporal_features = compute_temporal_features(transactions)
    all_customers = all_customers.merge(temporal_features, on='customer_id', how='left')

    print("   - unique_merchant_categories")
    card_data = data.get('card', pd.DataFrame())
    merchant_features = compute_unique_merchant_categories(card_data, all_customers['customer_id'])
    all_customers = all_customers.merge(merchant_features, on='customer_id', how='left')


    # Fill NaN values
    print("\n4. Handling missing values...")
    numeric_cols = all_customers.select_dtypes(include=[np.number]).columns
    all_customers[numeric_cols] = all_customers[numeric_cols].fillna(0)

    print(f"\n5. Final feature matrix: {all_customers.shape[0]} customers x {all_customers.shape[1]} columns")

    # Save output
    if output_path is None:
        output_path = os.path.join(data_path, 'yifei_aml_features_selected.parquet')

    all_customers.to_parquet(output_path, index=False)
    print(f"\n6. Features saved to: {output_path}")

    print("=" * 60)

    return all_customers


def main():
    parser = argparse.ArgumentParser(description='AML Feature Engineering')

    parser.add_argument(
        '--data_path',
        type=str,
        default='/Users/yc/Documents/GitHub/IMI_Case_Codes/',
        help='Path to directory containing parquet files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output parquet file path'
    )

    args = parser.parse_args()

    # Run feature engineering
    features = run_feature_engineering(args.data_path, args.output)

    # Display results
    print("\n" + "=" * 60)
    print("FEATURE COLUMNS")
    print("=" * 60)
    for col in features.columns:
        print(f"  - {col}")

    print("\n" + "=" * 60)
    print("SAMPLE DATA (first 10 rows)")
    print("=" * 60)
    print(features.head(10).to_string())

    print("\n" + "=" * 60)
    print("FEATURE STATISTICS")
    print("=" * 60)
    print(features.describe().round(2).to_string())


if __name__ == "__main__":
    main()