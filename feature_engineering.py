#5. Feature Engineering
# src/feature_engineering.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Create features for customer segmentation and predictive modeling"""

    def __init__(self):
        self.features = {}

    def load_data(self):
        """Load processed data"""
        try:
            self.customers = pd.read_csv('/content/customers_cleaned.csv')
            self.transactions = pd.read_csv('/content/transactions_cleaned.csv')
            self.interactions = pd.read_csv('/content/interactions_cleaned.csv')
            self.merged_data = pd.read_csv('/content/merged_data.csv')

            # Convert date columns
            if 'signup_date' in self.customers.columns:
                self.customers['signup_date'] = pd.to_datetime(self.customers['signup_date'])
            if 'transaction_date' in self.transactions.columns:
                self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'])
            if 'timestamp' in self.interactions.columns:
                self.interactions['timestamp'] = pd.to_datetime(self.interactions['timestamp'])
            if 'transaction_date' in self.merged_data.columns:
                self.merged_data['transaction_date'] = pd.to_datetime(self.merged_data['transaction_date'])

            print("Data loaded for feature engineering!")

        except Exception as e:
            print(f"Error loading data: {e}")
            # Create empty dataframes if files don't exist
            self.customers = pd.DataFrame()
            self.transactions = pd.DataFrame()
            self.interactions = pd.DataFrame()
            self.merged_data = pd.DataFrame()

    def find_financial_columns(self):
        """Find available financial columns in the dataset"""
        possible_revenue_cols = ['revenue', 'amount', 'price', 'total_amount', 'sales', 'transaction_amount']
        possible_quantity_cols = ['quantity', 'qty', 'units']

        # Find revenue column in transactions
        self.revenue_col = None
        if not self.transactions.empty:
            for col in possible_revenue_cols:
                if col in self.transactions.columns:
                    self.revenue_col = col
                    break

        # Find quantity column in transactions
        self.quantity_col = None
        if not self.transactions.empty:
            for col in possible_quantity_cols:
                if col in self.transactions.columns:
                    self.quantity_col = col
                    break

        print(f"Using revenue column: {self.revenue_col}")
        print(f"Using quantity column: {self.quantity_col}")

        return self.revenue_col, self.quantity_col

    def create_rfm_features(self):
        """Create Recency, Frequency, Monetary features"""
        print("Creating RFM features...")

        if self.transactions.empty:
            print("No transaction data available for RFM features")
            return pd.DataFrame()

        # Find financial columns
        revenue_col, quantity_col = self.find_financial_columns()

        if not revenue_col:
            print("No revenue column found for RFM features")
            return pd.DataFrame()

        try:
            # Calculate RFM metrics
            current_date = self.transactions['transaction_date'].max()

            rfm_agg = {
                'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
                'transaction_id': 'count'  # Frequency
            }

            # Add monetary aggregation if revenue column exists
            if revenue_col:
                rfm_agg[revenue_col] = 'sum'  # Monetary

            rfm = self.transactions.groupby('customer_id').agg(rfm_agg).reset_index()

            # Set column names based on available aggregations
            if revenue_col:
                rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
            else:
                rfm.columns = ['customer_id', 'recency', 'frequency']
                # If no monetary column, use frequency as proxy
                rfm['monetary'] = rfm['frequency']

            # Create RFM scores (handle cases with few unique values)
            try:
                rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
            except ValueError:
                # If not enough unique values, use rank-based approach
                rfm['recency_score'] = pd.cut(rfm['recency'], bins=min(5, len(rfm)), labels=range(min(5, len(rfm)), 0, -1))

            try:
                rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            except ValueError:
                rfm['frequency_score'] = pd.cut(rfm['frequency'], bins=min(5, len(rfm)), labels=range(1, min(6, len(rfm)+1)))

            try:
                rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            except ValueError:
                rfm['monetary_score'] = pd.cut(rfm['monetary'], bins=min(5, len(rfm)), labels=range(1, min(6, len(rfm)+1)))

            # Convert scores to numeric
            rfm['recency_score'] = rfm['recency_score'].astype(int)
            rfm['frequency_score'] = rfm['frequency_score'].astype(int)
            rfm['monetary_score'] = rfm['monetary_score'].astype(int)

            # Calculate RFM segment
            rfm['rfm_score'] = rfm['recency_score'] + rfm['frequency_score'] + rfm['monetary_score']

            # Define RFM segments
            def get_rfm_segment(score):
                if score >= 12:
                    return 'Champions'
                elif score >= 9:
                    return 'Loyal Customers'
                elif score >= 7:
                    return 'Potential Loyalists'
                elif score >= 5:
                    return 'At Risk'
                else:
                    return 'Cannot Lose'

            rfm['rfm_segment'] = rfm['rfm_score'].apply(get_rfm_segment)

            self.rfm_features = rfm
            print(f"RFM features created for {len(rfm)} customers")
            return rfm

        except Exception as e:
            print(f"Error creating RFM features: {e}")
            return pd.DataFrame()

    def create_behavioral_features(self):
        """Create behavioral features from website interactions"""
        print("Creating behavioral features...")

        if self.interactions.empty:
            print("No interaction data available for behavioral features")
            return pd.DataFrame()

        try:
            # Session-level features - only use numerical columns for aggregation
            behavioral_features = self.interactions.groupby('customer_id').agg({
                'session_duration': ['mean', 'max', 'sum'] if 'session_duration' in self.interactions.columns else 'count',
                'products_viewed': ['mean', 'max', 'sum'] if 'products_viewed' in self.interactions.columns else 'count',
                'interaction_id': 'count'
            }).reset_index()

            # Flatten column names
            behavioral_features.columns = [
                'customer_id', 'avg_session_duration', 'max_session_duration',
                'total_session_duration', 'avg_products_viewed', 'max_products_viewed',
                'total_products_viewed', 'total_interactions'
            ]

            # Page-specific features (only if columns exist)
            if 'page_visited' in self.interactions.columns and 'action' in self.interactions.columns:
                page_features = pd.get_dummies(
                    self.interactions,
                    columns=['page_visited', 'action'],
                    prefix=['page', 'action']
                )

                page_agg = page_features.groupby('customer_id').sum().reset_index()

                # Merge all behavioral features
                behavioral_features = behavioral_features.merge(
                    page_agg, on='customer_id', how='left'
                )
            else:
                print("Page visited or action columns not found in interactions data")

            self.behavioral_features = behavioral_features
            print(f"Behavioral features created for {len(behavioral_features)} customers")
            return behavioral_features

        except Exception as e:
            print(f"Error creating behavioral features: {e}")
            return pd.DataFrame()

    def create_temporal_features(self):
        """Create time-based features"""
        print("Creating temporal features...")

        if self.customers.empty or self.transactions.empty:
            print("No customer or transaction data available for temporal features")
            return pd.DataFrame()

        try:
            # Customer tenure
            current_date = self.transactions['transaction_date'].max()
            temporal_features = self.customers[['customer_id', 'signup_date']].copy()

            if 'signup_date' in temporal_features.columns:
                temporal_features['tenure_days'] = (
                    current_date - temporal_features['signup_date']
                ).dt.days

            # Purchase time patterns
            purchase_times = self.transactions.groupby('customer_id').agg({
                'transaction_date': ['min', 'max', 'nunique']
            }).reset_index()

            purchase_times.columns = [
                'customer_id', 'first_purchase', 'last_purchase', 'purchase_days'
            ]

            purchase_times['purchase_span_days'] = (
                purchase_times['last_purchase'] - purchase_times['first_purchase']
            ).dt.days

            # Merge temporal features
            temporal_features = temporal_features.merge(
                purchase_times, on='customer_id', how='left'
            )

            self.temporal_features = temporal_features
            return temporal_features

        except Exception as e:
            print(f"Error creating temporal features: {e}")
            return pd.DataFrame()

    def create_demographic_features(self):
        """Create and encode demographic features"""
        print("Creating demographic features...")

        if self.customers.empty:
            print("No customer data available for demographic features")
            return pd.DataFrame()

        try:
            # Select available demographic columns
            available_cols = ['customer_id']
            if 'age' in self.customers.columns:
                available_cols.append('age')
            if 'gender' in self.customers.columns:
                available_cols.append('gender')
            if 'country' in self.customers.columns:
                available_cols.append('country')
            if 'income_segment' in self.customers.columns:
                available_cols.append('income_segment')

            demographic = self.customers[available_cols].copy()

            # One-hot encoding for categorical variables
            categorical_cols = []
            if 'gender' in demographic.columns:
                categorical_cols.append('gender')
            if 'country' in demographic.columns:
                categorical_cols.append('country')
            if 'income_segment' in demographic.columns:
                categorical_cols.append('income_segment')

            if categorical_cols:
                demographic_encoded = pd.get_dummies(
                    demographic,
                    columns=categorical_cols,
                    prefix=categorical_cols
                )
            else:
                demographic_encoded = demographic.copy()

            # Age groups (if age column exists)
            if 'age' in demographic.columns:
                def get_age_group(age):
                    if age <= 25:
                        return '18-25'
                    elif age <= 35:
                        return '26-35'
                    elif age <= 45:
                        return '36-45'
                    elif age <= 55:
                        return '46-55'
                    else:
                        return '55+'

                demographic['age_group'] = demographic['age'].apply(get_age_group)
                age_encoded = pd.get_dummies(demographic['age_group'], prefix='age')
                demographic_encoded = pd.concat([demographic_encoded, age_encoded], axis=1)

            self.demographic_features = demographic_encoded
            return demographic_encoded

        except Exception as e:
            print(f"Error creating demographic features: {e}")
            return pd.DataFrame()

    def create_product_preference_features(self):
        """Create features related to product preferences"""
        print("Creating product preference features...")

        if self.merged_data.empty:
            print("No merged data available for product preference features")
            return pd.DataFrame()

        try:
            # Find financial columns in merged data
            revenue_col, quantity_col = self.find_financial_columns()

            if not revenue_col:
                print("No revenue column found for product preference features")
                return pd.DataFrame()

            # Category preferences
            category_pref = self.merged_data.groupby(['customer_id', 'category']).agg({
                revenue_col: 'sum',
            }).reset_index()

            if quantity_col and quantity_col in self.merged_data.columns:
                quantity_agg = self.merged_data.groupby(['customer_id', 'category'])[quantity_col].sum().reset_index()
                category_pref = category_pref.merge(quantity_agg, on=['customer_id', 'category'])

            # Pivot to get category spending
            category_pivot = category_pref.pivot_table(
                index='customer_id',
                columns='category',
                values=revenue_col,
                fill_value=0
            ).reset_index()

            category_pivot.columns = [f'category_{col}_revenue' if col != 'customer_id' else col
                                    for col in category_pivot.columns]

            # Favorite category
            favorite_category = category_pref.loc[
                category_pref.groupby('customer_id')[revenue_col].idxmax()
            ][['customer_id', 'category']]
            favorite_category.columns = ['customer_id', 'favorite_category']

            # Price sensitivity (if price column exists in merged data)
            price_features = pd.DataFrame()
            if 'price' in self.merged_data.columns:
                price_sensitivity = self.merged_data.groupby('customer_id').agg({
                    'price': ['mean', 'std', 'min', 'max']
                }).reset_index()

                price_sensitivity.columns = [
                    'customer_id', 'avg_product_price', 'price_std',
                    'min_product_price', 'max_product_price'
                ]
                price_features = price_sensitivity

            # Merge all product preference features
            product_features = category_pivot.merge(
                favorite_category, on='customer_id', how='left'
            )

            if not price_features.empty:
                product_features = product_features.merge(
                    price_features, on='customer_id', how='left'
                )

            self.product_features = product_features
            return product_features

        except Exception as e:
            print(f"Error creating product preference features: {e}")
            return pd.DataFrame()

    def create_engineered_dataset(self):
        """Combine all features into a comprehensive dataset"""
        print("Creating comprehensive feature dataset...")

        # Check if we have at least RFM features
        if not hasattr(self, 'rfm_features') or self.rfm_features.empty:
            print("No RFM features available. Cannot create comprehensive dataset.")
            return pd.DataFrame()

        try:
            # Start with RFM features
            comprehensive_features = self.rfm_features.copy()

            # List of all available feature sets to merge
            feature_sets_to_merge = []

            if hasattr(self, 'behavioral_features') and not self.behavioral_features.empty:
                feature_sets_to_merge.append(self.behavioral_features)

            if hasattr(self, 'temporal_features') and not self.temporal_features.empty:
                # Drop date columns before merging
                cols_to_drop = [col for col in ['signup_date', 'first_purchase', 'last_purchase']
                              if col in self.temporal_features.columns]
                temporal_for_merge = self.temporal_features.drop(cols_to_drop, axis=1)
                feature_sets_to_merge.append(temporal_for_merge)

            if hasattr(self, 'demographic_features') and not self.demographic_features.empty:
                # Drop original age column if exists
                cols_to_drop = [col for col in ['age'] if col in self.demographic_features.columns]
                demographic_for_merge = self.demographic_features.drop(cols_to_drop, axis=1)
                feature_sets_to_merge.append(demographic_for_merge)

            if hasattr(self, 'product_features') and not self.product_features.empty:
                feature_sets_to_merge.append(self.product_features)

            # Merge all feature sets
            for feature_set in feature_sets_to_merge:
                comprehensive_features = comprehensive_features.merge(
                    feature_set, on='customer_id', how='left'
                )

            # Handle missing values
            comprehensive_features = comprehensive_features.fillna(0)

            # Remove duplicate columns
            comprehensive_features = comprehensive_features.loc[:, ~comprehensive_features.columns.duplicated()]

            print(f"Comprehensive feature dataset created with {len(comprehensive_features)} customers and {len(comprehensive_features.columns)} features")
            return comprehensive_features

        except Exception as e:
            print(f"Error creating comprehensive dataset: {e}")
            return pd.DataFrame()

    def prepare_modeling_data(self, comprehensive_features):
        """Prepare data for machine learning modeling"""
        print("Preparing data for modeling...")

        if comprehensive_features.empty:
            print("No features available for modeling")
            return {}

        try:
            # Separate features and target (if any)
            feature_columns = [col for col in comprehensive_features.columns
                             if col not in ['customer_id', 'rfm_segment']]

            X = comprehensive_features[feature_columns]

            # Scale numerical features
            scaler = StandardScaler()
            numerical_columns = X.select_dtypes(include=[np.number]).columns

            # Only scale if we have numerical columns
            if len(numerical_columns) > 0:
                X_scaled = X.copy()
                X_scaled[numerical_columns] = scaler.fit_transform(X[numerical_columns])
            else:
                X_scaled = X.copy()

            # Save scaler for future use
            self.scaler = scaler

            modeling_data = {
                'X': X,
                'X_scaled': X_scaled,
                'customer_ids': comprehensive_features['customer_id'],
                'feature_names': feature_columns
            }

            # Add RFM segments if available
            if 'rfm_segment' in comprehensive_features.columns:
                modeling_data['rfm_segments'] = comprehensive_features['rfm_segment']

            return modeling_data

        except Exception as e:
            print(f"Error preparing modeling data: {e}")
            return {}

    def run_feature_engineering_pipeline(self):
        """Execute complete feature engineering pipeline"""
        print("Starting Feature Engineering Pipeline...")

        self.load_data()

        # Only proceed if we have transaction data
        if self.transactions.empty:
            print("No transaction data available. Cannot create features.")
            return {}

        self.create_rfm_features()
        self.create_behavioral_features()
        self.create_temporal_features()
        self.create_demographic_features()
        self.create_product_preference_features()

        comprehensive_features = self.create_engineered_dataset()

        if comprehensive_features.empty:
            print("Failed to create comprehensive features dataset")
            return {}

        modeling_data = self.prepare_modeling_data(comprehensive_features)

        # Save engineered features
        try:
            comprehensive_features.to_csv('/content/engineered_features.csv', index=False)
            print("Engineered features saved to '/content/engineered_features.csv'")
        except Exception as e:
            print(f"Error saving engineered features: {e}")

        print("\nFeature engineering completed!")
        if comprehensive_features is not None:
            print(f"Final dataset shape: {comprehensive_features.shape}")

        return modeling_data

if __name__ == "__main__":
    engineer = FeatureEngineer()
    modeling_data = engineer.run_feature_engineering_pipeline()