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
            data[file] = pd.read_csv(f'{data_path}/{file}.csv')
            print(f"Loaded {file}.csv: {len(data[file]):,} rows")
        except FileNotFoundError:
            print(f"Warning: {file}.csv not found")

    # KYC files
    kyc_files = ['kyc_individual', 'kyc_smallbusiness', 'kyc_occupation_codes', 'kyc_industry_codes']
    for file in kyc_files:
        try:
            data[file] = pd.read_csv(f'{data_path}/{file}.csv')
            print(f"Loaded {file}.csv: {len(data[file]):,} rows")
        except FileNotFoundError:
            print(f"Warning: {file}.csv not found")

    # Labels
    try:
        data['labels'] = pd.read_csv(f'{data_path}/labels.csv')
        print(f"Loaded labels.csv: {len(data['labels']):,} rows")
    except FileNotFoundError:
        print("Warning: labels.csv not found")

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


def compute_num_channels_used(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Feature: num_channels_used
    Count of unique transaction channels used by each customer.
    """
    features = transactions.groupby('customer_id')['channel'].nunique().reset_index()
    features.columns = ['customer_id', 'num_channels_used']
    return features


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


def compute_geographic_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Features: unique_provinces, unique_countries, unique_cities, geo_diversity_score
    """
    # Filter transactions with location data (ABM, Card)
    geo_txn = transactions[transactions['country'].notna()].copy()

    # Get all customer IDs
    all_customers = pd.DataFrame({'customer_id': transactions['customer_id'].unique()})

    if len(geo_txn) == 0:
        all_customers['unique_countries'] = 0
        all_customers['unique_provinces'] = 0
        all_customers['unique_cities'] = 0
        all_customers['geo_diversity_score'] = 0
        return all_customers

    geo_features = geo_txn.groupby('customer_id').agg({
        'country': 'nunique',
        'province': 'nunique',
        'city': 'nunique',
    })
    geo_features.columns = ['unique_countries', 'unique_provinces', 'unique_cities']

    # geo_diversity_score
    geo_features['geo_diversity_score'] = (
            geo_features['unique_countries'] * 10 +
            geo_features['unique_provinces'] * 2 +
            geo_features['unique_cities']
    )

    # Merge with all customers
    all_customers = all_customers.merge(geo_features.reset_index(), on='customer_id', how='left')
    all_customers = all_customers.fillna(0)

    return all_customers


def compute_near_threshold_count(transactions: pd.DataFrame, threshold: float = 10000) -> pd.DataFrame:
    """
    Feature: near_threshold_count
    Count of transactions within 10% below the reporting threshold.
    """
    features_list = []

    for customer_id, group in transactions.groupby('customer_id'):
        amounts = group['amount_cad'].values
        # Transactions within 90-100% of threshold
        near_threshold = ((threshold * 0.9 <= amounts) & (amounts < threshold)).sum()

        features_list.append({
            'customer_id': customer_id,
            'near_threshold_count': near_threshold
        })

    return pd.DataFrame(features_list)


def compute_avg_txn_per_active_day(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Feature: avg_txn_per_active_day
    Average number of transactions per day when customer was active.
    """
    features = transactions.groupby('customer_id').agg({
        'transaction_id': 'count',
        'date': 'nunique'
    })
    features.columns = ['total_txn', 'unique_days']

    features['avg_txn_per_active_day'] = features['total_txn'] / features['unique_days']

    return features[['avg_txn_per_active_day']].reset_index()


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


def compute_txn_to_income_ratio(transactions: pd.DataFrame,
                                kyc_individual: pd.DataFrame,
                                kyc_smallbusiness: pd.DataFrame) -> pd.DataFrame:
    """
    Feature: txn_to_income_ratio
    Ratio of total transaction amount to stated income/sales.
    """
    # Get total transaction amounts per customer
    txn_totals = transactions.groupby('customer_id')['amount_cad'].sum().reset_index()
    txn_totals.columns = ['customer_id', 'total_txn_amount']

    # Get income from individual KYC
    individual_income = kyc_individual[['customer_id', 'income']].copy()

    # Get sales from business KYC (use as income proxy)
    business_income = kyc_smallbusiness[['customer_id', 'sales']].copy()
    business_income.columns = ['customer_id', 'income']

    # Combine
    all_income = pd.concat([individual_income, business_income], ignore_index=True)

    # Merge with transaction totals
    features = txn_totals.merge(all_income, on='customer_id', how='left')

    # Calculate ratio (avoid division by zero)
    features['txn_to_income_ratio'] = features['total_txn_amount'] / features['income'].replace(0, 1)
    features.loc[features['income'] == 0, 'txn_to_income_ratio'] = 0

    return features[['customer_id', 'txn_to_income_ratio']]


def run_feature_engineering(data_path: str = './') -> pd.DataFrame:
    """
    Main function to run feature engineering for selected features only.
    """
    print("=" * 60)
    print("AML Feature Engineering - Selected Features")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    data = load_all_data(data_path)

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

    print("   - num_channels_used")
    channels_features = compute_num_channels_used(transactions)
    all_customers = all_customers.merge(channels_features, on='customer_id', how='left')

    print("   - late_night_ratio, late_night_txn_count, weekend_txn_count, activity_span_days")
    temporal_features = compute_temporal_features(transactions)
    all_customers = all_customers.merge(temporal_features, on='customer_id', how='left')

    print("   - unique_provinces, unique_countries, unique_cities, geo_diversity_score")
    geo_features = compute_geographic_features(transactions)
    all_customers = all_customers.merge(geo_features, on='customer_id', how='left')

    print("   - near_threshold_count")
    threshold_features = compute_near_threshold_count(transactions)
    all_customers = all_customers.merge(threshold_features, on='customer_id', how='left')

    print("   - avg_txn_per_active_day")
    active_day_features = compute_avg_txn_per_active_day(transactions)
    all_customers = all_customers.merge(active_day_features, on='customer_id', how='left')

    print("   - unique_merchant_categories")
    card_data = data.get('card', pd.DataFrame())
    merchant_features = compute_unique_merchant_categories(card_data, all_customers['customer_id'])
    all_customers = all_customers.merge(merchant_features, on='customer_id', how='left')

    print("   - txn_to_income_ratio")
    if 'kyc_individual' in data and 'kyc_smallbusiness' in data:
        ratio_features = compute_txn_to_income_ratio(
            transactions, data['kyc_individual'], data['kyc_smallbusiness']
        )
        all_customers = all_customers.merge(ratio_features, on='customer_id', how='left')

    # Add labels if available
    if 'labels' in data:
        print("   - Adding labels")
        all_customers = all_customers.merge(data['labels'], on='customer_id', how='left')

    # Fill NaN values
    print("\n4. Handling missing values...")
    numeric_cols = all_customers.select_dtypes(include=[np.number]).columns
    all_customers[numeric_cols] = all_customers[numeric_cols].fillna(0)

    print(f"\n5. Final feature matrix: {all_customers.shape[0]} customers x {all_customers.shape[1]} columns")
    print("=" * 60)

    return all_customers


def create_synthetic_data() -> Dict[str, pd.DataFrame]:
    """
    Create synthetic data for demonstration purposes.
    """
    np.random.seed(42)

    # Customer IDs
    individual_ids = [f'IND_{i:05d}' for i in range(800)]
    business_ids = [f'BUS_{i:05d}' for i in range(200)]
    all_customer_ids = individual_ids + business_ids

    def create_transactions(channel, n, has_location=False, has_merchant=False):
        df = pd.DataFrame({
            'transaction_id': [f'{channel}_{i:07d}' for i in range(n)],
            'customer_id': np.random.choice(all_customer_ids, n),
            'amount_cad': np.random.exponential(500, n),
            'debit_credit': np.random.choice(['D', 'C'], n, p=[0.55, 0.45]),
            'transaction_datetime': pd.date_range(
                '2024-01-01', periods=n, freq='1min'
            ) + pd.to_timedelta(np.random.randint(0, 365 * 24 * 60, n), unit='min')
        })

        if has_location:
            df['country'] = np.random.choice(['CA', 'US', 'MX', 'UK'], n, p=[0.85, 0.10, 0.03, 0.02])
            df['province'] = np.random.choice(['ON', 'BC', 'AB', 'QC'], n)
            df['city'] = np.random.choice(['Toronto', 'Vancouver', 'Calgary', 'Montreal', 'Ottawa'], n)

        if has_merchant:
            df['merchant_category'] = np.random.choice(
                ['Retail', 'Restaurant', 'Gas', 'Financial Services', 'Entertainment'],
                n, p=[0.4, 0.2, 0.15, 0.15, 0.1]
            )

        return df

    data = {
        'abm': create_transactions('ABM', 8000, has_location=True),
        'card': create_transactions('CARD', 20000, has_location=True, has_merchant=True),
        'cheque': create_transactions('CHQ', 3000),
        'eft': create_transactions('EFT', 10000),
        'emt': create_transactions('EMT', 5000),
        'westernunion': create_transactions('WU', 2000),
        'wire': create_transactions('WIRE', 2000),
    }

    data['kyc_individual'] = pd.DataFrame({
        'customer_id': individual_ids,
        'income': np.random.lognormal(10.5, 0.8, len(individual_ids)),
    })

    data['kyc_smallbusiness'] = pd.DataFrame({
        'customer_id': business_ids,
        'sales': np.random.lognormal(13, 1.5, len(business_ids)),
    })

    data['labels'] = pd.DataFrame({
        'customer_id': all_customer_ids,
        'label': np.random.choice([0, 1], len(all_customer_ids), p=[0.9, 0.1])
    })

    return data


if __name__ == "__main__":
    # Run demo with synthetic data
    print("Creating synthetic data for demonstration...")
    synthetic_data = create_synthetic_data()

    # Save synthetic data to temp files
    import os

    temp_dir = '/tmp/aml_data'
    os.makedirs(temp_dir, exist_ok=True)

    for name, df in synthetic_data.items():
        df.to_csv(f'{temp_dir}/{name}.csv', index=False)

    # Run feature engineering
    features = run_feature_engineering(temp_dir)

    # Display results
    print("\n" + "=" * 60)
    print("SELECTED FEATURES")
    print("=" * 60)
    print(f"\nFeature columns:")
    for col in features.columns:
        print(f"  - {col}")

    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    print(features.head(10).to_string())

    # Summary statistics
    print("\n" + "=" * 60)
    print("FEATURE STATISTICS")
    print("=" * 60)
    print(features.describe().round(2).to_string())

    # Save features
    features.to_csv('./aml_features_selected.csv', index=False)
    print(f"\nFeatures saved to: output_path")
