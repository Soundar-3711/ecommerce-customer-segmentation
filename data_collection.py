#2. Data Collection Module

# src/data_collection.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataGenerator:
    """Generate synthetic e-commerce data for analysis"""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Sample data for generation
        self.products = [
            'Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch',
            'Camera', 'Gaming Console', 'Monitor', 'Keyboard', 'Mouse'
        ]

        self.categories = ['Electronics', 'Computers', 'Mobile', 'Gaming', 'Accessories']
        self.countries = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP']

    def generate_customers(self, n_customers=1000):
        """Generate customer demographic data"""
        customers = []

        for i in range(n_customers):
            customer = {
                'customer_id': f'CUST_{i:04d}',
                'age': np.random.randint(18, 70),
                'gender': np.random.choice(['Male', 'Female'], p=[0.48, 0.52]),
                'country': np.random.choice(self.countries),
                'signup_date': datetime(2022, 1, 1) + timedelta(
                    days=np.random.randint(0, 365)
                ),
                'income_segment': np.random.choice(
                    ['Low', 'Medium', 'High'],
                    p=[0.3, 0.5, 0.2]
                )
            }
            customers.append(customer)

        return pd.DataFrame(customers)

    def generate_products(self):
        """Generate product catalog"""
        products = []

        for i, product in enumerate(self.products):
            product_data = {
                'product_id': f'PROD_{i:03d}',
                'product_name': product,
                'category': np.random.choice(self.categories),
                'price': np.random.uniform(50, 2000),
                'cost': np.random.uniform(20, 1500)
            }
            products.append(product_data)

        return pd.DataFrame(products)

    def generate_transactions(self, customers_df, products_df, n_transactions=10000):
        """Generate transaction data"""
        transactions = []

        for i in range(n_transactions):
            customer = customers_df.sample(1).iloc[0]
            product = products_df.sample(1).iloc[0]

            transaction_date = customer['signup_date'] + timedelta(
                days=np.random.randint(0, 365)
            )

            # Ensure transaction date is within reasonable range
            if transaction_date > datetime(2023, 12, 31):
                continue

            quantity = np.random.randint(1, 4)
            revenue = product['price'] * quantity
            profit = (product['price'] - product['cost']) * quantity

            transaction = {
                'transaction_id': f'TRX_{i:05d}',
                'customer_id': customer['customer_id'],
                'product_id': product['product_id'],
                'transaction_date': transaction_date,
                'quantity': quantity,
                'revenue': revenue,
                'profit': profit,
                'payment_method': np.random.choice(
                    ['Credit Card', 'PayPal', 'Bank Transfer', 'Cash']
                )
            }
            transactions.append(transaction)

        return pd.DataFrame(transactions)

    def generate_website_interactions(self, customers_df, n_interactions=50000):
        """Generate website interaction data"""
        interactions = []
        pages = ['homepage', 'product_page', 'cart', 'checkout', 'category_page']
        actions = ['view', 'click', 'add_to_cart', 'purchase', 'search']

        for i in range(n_interactions):
            customer = customers_df.sample(1).iloc[0]

            interaction = {
                'interaction_id': f'INT_{i:06d}',
                'customer_id': customer['customer_id'],
                'timestamp': datetime(2023, 1, 1) + timedelta(
                    seconds=np.random.randint(0, 365*24*60*60)
                ),
                'page_visited': np.random.choice(pages),
                'action': np.random.choice(actions),
                'session_duration': np.random.exponential(300),  # seconds
                'products_viewed': np.random.randint(1, 10)
            }
            interactions.append(interaction)

        return pd.DataFrame(interactions)

def collect_data():
    """Main function to generate all datasets"""
    generator = DataGenerator()

    print("Generating customer data...")
    customers = generator.generate_customers(1500)

    print("Generating product data...")
    products = generator.generate_products()

    print("Generating transaction data...")
    transactions = generator.generate_transactions(customers, products, 15000)

    print("Generating website interaction data...")
    interactions = generator.generate_website_interactions(customers, 75000)

    return customers, products, transactions, interactions

if __name__ == "__main__":
    customers, products, transactions, interactions = collect_data()

    # Save raw data
    customers.to_csv('/content/customers.csv', index=False)
    products.to_csv('/content/products.csv', index=False)
    transactions.to_csv('/content/transactions.csv', index=False)
    interactions.to_csv('/content/interactions.csv', index=False)

    print("Data generation completed!")
    print(f"Customers: {len(customers)}")
    print(f"Products: {len(products)}")
    print(f"Transactions: {len(transactions)}")
    print(f"Interactions: {len(interactions)}")