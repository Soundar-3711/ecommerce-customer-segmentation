#3. Data Cleaning and Preprocessing
# src/data_cleaning.py
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Clean and preprocess e-commerce data"""

    def __init__(self):
        self.processed_data = {}

    def load_raw_data(self):
        """Load raw data files"""
        try:
            self.customers = pd.read_csv('/content/customers.csv')
            self.products = pd.read_csv('/content/products.csv')
            self.transactions = pd.read_csv('/content/transactions.csv')
            self.interactions = pd.read_csv('/content/interactions.csv')
            print("Raw data loaded successfully!")
        except FileNotFoundError:
            print("Raw data files not found. Please run data collection first.")
            raise

    def clean_customer_data(self):
        """Clean customer demographic data"""
        print("Cleaning customer data...")

        # Convert date columns
        self.customers['signup_date'] = pd.to_datetime(self.customers['signup_date'])

        # Handle missing values
        self.customers['income_segment'] = self.customers['income_segment'].fillna('Unknown')

        # Remove duplicates
        self.customers = self.customers.drop_duplicates(subset=['customer_id'])

        # Validate age range
        self.customers = self.customers[
            (self.customers['age'] >= 18) & (self.customers['age'] <= 100)
        ]

        print(f"Customer data cleaned: {len(self.customers)} records")
        return self.customers

    def clean_product_data(self):
        """Clean product data"""
        print("Cleaning product data...")

        # Remove duplicates
        self.products = self.products.drop_duplicates(subset=['product_id'])

        # Validate price and cost
        self.products = self.products[
            (self.products['price'] > 0) &
            (self.products['cost'] > 0) &
            (self.products['price'] >= self.products['cost'])
        ]

        # Fill missing categories
        self.products['category'] = self.products['category'].fillna('Unknown')

        print(f"Product data cleaned: {len(self.products)} records")
        return self.products

    def clean_transaction_data(self):
        """Clean transaction data"""
        print("Cleaning transaction data...")

        # Convert date columns
        self.transactions['transaction_date'] = pd.to_datetime(
            self.transactions['transaction_date']
        )

        # Remove invalid transactions
        self.transactions = self.transactions[
            (self.transactions['quantity'] > 0) &
            (self.transactions['revenue'] > 0) &
            (self.transactions['profit'].notna())
        ]

        # Remove duplicates
        self.transactions = self.transactions.drop_duplicates(
            subset=['transaction_id']
        )

        # Filter transactions within valid date range
        start_date = pd.Timestamp('2022-01-01')
        end_date = pd.Timestamp('2023-12-31')
        self.transactions = self.transactions[
            (self.transactions['transaction_date'] >= start_date) &
            (self.transactions['transaction_date'] <= end_date)
        ]

        print(f"Transaction data cleaned: {len(self.transactions)} records")
        return self.transactions

    def clean_interaction_data(self):
        """Clean website interaction data"""
        print("Cleaning interaction data...")

        # Convert timestamp
        self.interactions['timestamp'] = pd.to_datetime(
            self.interactions['timestamp']
        )

        # Remove invalid interactions
        self.interactions = self.interactions[
            (self.interactions['session_duration'] >= 0) &
            (self.interactions['products_viewed'] >= 0)
        ]

        # Remove duplicates
        self.interactions = self.interactions.drop_duplicates(
            subset=['interaction_id']
        )

        # Filter interactions within valid date range
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-12-31')
        self.interactions = self.interactions[
            (self.interactions['timestamp'] >= start_date) &
            (self.interactions['timestamp'] <= end_date)
        ]

        print(f"Interaction data cleaned: {len(self.interactions)} records")
        return self.interactions

    def create_merged_dataset(self):
        """Create a comprehensive merged dataset for analysis"""
        print("Creating merged dataset...")

        # Merge transactions with customer and product data
        merged_data = self.transactions.merge(
            self.customers, on='customer_id', how='left'
        ).merge(
            self.products, on='product_id', how='left'
        )

        # Calculate additional features
        merged_data['profit_margin'] = (
            merged_data['profit'] / merged_data['revenue']
        ) * 100

        merged_data['month'] = merged_data['transaction_date'].dt.month
        merged_data['quarter'] = merged_data['transaction_date'].dt.quarter
        merged_data['year'] = merged_data['transaction_date'].dt.year

        print(f"Merged dataset created: {len(merged_data)} records")
        return merged_data

    def save_cleaned_data(self):
        """Save all cleaned datasets"""
        self.customers.to_csv('/content/customers_cleaned.csv', index=False)
        self.products.to_csv('/content/products_cleaned.csv', index=False)
        self.transactions.to_csv('/content/transactions_cleaned.csv', index=False)
        self.interactions.to_csv('/content/interactions_cleaned.csv', index=False)

        merged_data = self.create_merged_dataset()
        merged_data.to_csv('/content/merged_data.csv', index=False)

        print("All cleaned data saved successfully!")

    def run_cleaning_pipeline(self):
        """Execute complete data cleaning pipeline"""
        print("Starting data cleaning pipeline...")

        self.load_raw_data()
        self.clean_customer_data()
        self.clean_product_data()
        self.clean_transaction_data()
        self.clean_interaction_data()
        self.save_cleaned_data()

        print("Data cleaning pipeline completed!")

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run_cleaning_pipeline()