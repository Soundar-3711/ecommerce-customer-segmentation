#4. Exploratory Data Analysis
# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class EDA:
    """Perform exploratory data analysis on e-commerce data"""

    def __init__(self):
        self.data = {}
        self.figures = {}

    def load_cleaned_data(self):
        """Load cleaned datasets"""
        try:
            self.customers = pd.read_csv('/content/customers_cleaned.csv')
            self.products = pd.read_csv('/content/products_cleaned.csv')
            self.transactions = pd.read_csv('/content/transactions_cleaned.csv')
            self.interactions = pd.read_csv('/content/interactions_cleaned.csv')
            self.merged_data = pd.read_csv('/content/merged_data.csv')

            # Debug: Print column names to identify available columns
            print("Available columns in merged_data:", list(self.merged_data.columns))
            print("Available columns in transactions:", list(self.transactions.columns))

            # Convert date columns with error handling
            if 'signup_date' in self.customers.columns:
                self.customers['signup_date'] = pd.to_datetime(self.customers['signup_date'])

            if 'transaction_date' in self.transactions.columns:
                self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'])

            if 'timestamp' in self.interactions.columns:
                self.interactions['timestamp'] = pd.to_datetime(self.interactions['timestamp'])

            if 'transaction_date' in self.merged_data.columns:
                self.merged_data['transaction_date'] = pd.to_datetime(self.merged_data['transaction_date'])

            print("Cleaned data loaded successfully!")

        except Exception as e:
            print(f"Error loading data: {e}")
            # Create empty dataframes if files don't exist
            self.customers = pd.DataFrame()
            self.products = pd.DataFrame()
            self.transactions = pd.DataFrame()
            self.interactions = pd.DataFrame()
            self.merged_data = pd.DataFrame()

    def basic_statistics(self):
        """Generate basic statistics for all datasets"""
        print("=" * 50)
        print("BASIC DATASET STATISTICS")
        print("=" * 50)

        datasets = {
            'Customers': self.customers,
            'Products': self.products,
            'Transactions': self.transactions,
            'Interactions': self.interactions,
            'Merged Data': self.merged_data
        }

        for name, dataset in datasets.items():
            if not dataset.empty:
                print(f"\n{name}:")
                print(f"  Shape: {dataset.shape}")
                print(f"  Columns: {list(dataset.columns)}")
                print(f"  Missing values: {dataset.isnull().sum().sum()}")
            else:
                print(f"\n{name}: No data available")

    def find_financial_columns(self):
        """Find available financial columns in the dataset"""
        financial_columns = []
        possible_revenue_cols = ['revenue', 'amount', 'price', 'total_amount', 'sales']
        possible_profit_cols = ['profit', 'profit_amount', 'margin']
        possible_quantity_cols = ['quantity', 'qty', 'units']

        if not self.merged_data.empty:
            available_cols = self.merged_data.columns.tolist()
            print(f"Available columns: {available_cols}")

            # Find revenue column
            self.revenue_col = None
            for col in possible_revenue_cols:
                if col in available_cols:
                    self.revenue_col = col
                    break

            # Find profit column
            self.profit_col = None
            for col in possible_profit_cols:
                if col in available_cols:
                    self.profit_col = col
                    break

            # Find quantity column
            self.quantity_col = None
            for col in possible_quantity_cols:
                if col in available_cols:
                    self.quantity_col = col
                    break

            print(f"Using revenue column: {self.revenue_col}")
            print(f"Using profit column: {self.profit_col}")
            print(f"Using quantity column: {self.quantity_col}")

            return self.revenue_col, self.profit_col, self.quantity_col
        else:
            print("No merged data available")
            return None, None, None

    def customer_demographics_analysis(self):
        """Analyze customer demographics"""
        if self.customers.empty:
            print("No customer data available for demographics analysis")
            return

        print("\n" + "=" * 50)
        print("CUSTOMER DEMOGRAPHICS ANALYSIS")
        print("=" * 50)

        plt.figure(figsize=(12, 8))

        # Age distribution
        if 'age' in self.customers.columns:
            plt.subplot(2, 2, 1)
            sns.histplot(data=self.customers, x='age', bins=20, kde=True)
            plt.title('Age Distribution of Customers')
            plt.xlabel('Age')
            plt.ylabel('Count')

        # Gender distribution
        if 'gender' in self.customers.columns:
            plt.subplot(2, 2, 2)
            gender_counts = self.customers['gender'].value_counts()
            plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
            plt.title('Gender Distribution')

        # Country distribution
        if 'country' in self.customers.columns:
            plt.subplot(2, 2, 3)
            country_counts = self.customers['country'].value_counts().head(10)
            sns.barplot(x=country_counts.values, y=country_counts.index)
            plt.title('Top 10 Countries by Customer Count')
            plt.xlabel('Number of Customers')

        # Income segment distribution
        if 'income_segment' in self.customers.columns:
            plt.subplot(2, 2, 4)
            income_counts = self.customers['income_segment'].value_counts()
            sns.barplot(x=income_counts.values, y=income_counts.index)
            plt.title('Income Segment Distribution')
            plt.xlabel('Number of Customers')

        plt.tight_layout()
        plt.savefig('/content/customer_demographics.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary statistics
        if 'age' in self.customers.columns:
            print(f"Average customer age: {self.customers['age'].mean():.1f}")
        if 'gender' in self.customers.columns:
            print(f"Gender distribution:\n{self.customers['gender'].value_counts()}")
        if 'country' in self.customers.columns:
            print(f"Top countries:\n{self.customers['country'].value_counts().head()}")

    def sales_analysis(self):
        """Analyze sales patterns and trends"""
        if self.merged_data.empty:
            print("No merged data available for sales analysis")
            return

        print("\n" + "=" * 50)
        print("SALES ANALYSIS")
        print("=" * 50)

        # Find financial columns
        revenue_col, profit_col, quantity_col = self.find_financial_columns()

        if not revenue_col:
            print("No revenue column found for sales analysis")
            return

        # Monthly sales trend
        try:
            monthly_sales = self.merged_data.groupby(
                pd.Grouper(key='transaction_date', freq='M')
            ).agg({
                revenue_col: 'sum',
                'transaction_id': 'count'
            }).reset_index()

            if profit_col:
                monthly_profit = self.merged_data.groupby(
                    pd.Grouper(key='transaction_date', freq='M')
                )[profit_col].sum().reset_index()
                monthly_sales = monthly_sales.merge(monthly_profit, on='transaction_date')
                monthly_sales.columns = ['month', 'total_revenue', 'transaction_count', 'total_profit']
            else:
                monthly_sales.columns = ['month', 'total_revenue', 'transaction_count']

            plt.figure(figsize=(15, 10))

            # Revenue trend
            plt.subplot(2, 2, 1)
            plt.plot(monthly_sales['month'], monthly_sales['total_revenue'],
                    marker='o', linewidth=2, markersize=6)
            plt.title('Monthly Revenue Trend')
            plt.xlabel('Month')
            plt.ylabel('Revenue ($)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            # Profit trend (if available)
            if 'total_profit' in monthly_sales.columns:
                plt.subplot(2, 2, 2)
                plt.plot(monthly_sales['month'], monthly_sales['total_profit'],
                        marker='o', linewidth=2, markersize=6, color='green')
                plt.title('Monthly Profit Trend')
                plt.xlabel('Month')
                plt.ylabel('Profit ($)')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)

            # Transaction count trend
            plt.subplot(2, 2, 3)
            plt.plot(monthly_sales['month'], monthly_sales['transaction_count'],
                    marker='o', linewidth=2, markersize=6, color='orange')
            plt.title('Monthly Transaction Count Trend')
            plt.xlabel('Month')
            plt.ylabel('Number of Transactions')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            # Product category performance
            if 'category' in self.merged_data.columns:
                plt.subplot(2, 2, 4)
                category_revenue = self.merged_data.groupby('category')[revenue_col].sum().sort_values(ascending=False)
                sns.barplot(x=category_revenue.values, y=category_revenue.index)
                plt.title('Revenue by Product Category')
                plt.xlabel('Total Revenue ($)')

            plt.tight_layout()
            plt.savefig('/content/sales_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

            # Print sales insights
            print(f"Total Revenue: ${self.merged_data[revenue_col].sum():,.2f}")
            if profit_col:
                print(f"Total Profit: ${self.merged_data[profit_col].sum():,.2f}")
            print(f"Average Transaction Value: ${self.merged_data[revenue_col].mean():.2f}")
            if 'category' in self.merged_data.columns:
                print(f"Most profitable category: {category_revenue.index[0]} (${category_revenue.iloc[0]:,.2f})")

        except Exception as e:
            print(f"Error in sales analysis: {e}")

    def customer_behavior_analysis(self):
        """Analyze customer behavior patterns"""
        if self.merged_data.empty:
            print("No merged data available for customer behavior analysis")
            return

        print("\n" + "=" * 50)
        print("CUSTOMER BEHAVIOR ANALYSIS")
        print("=" * 50)

        # Find financial columns
        revenue_col, profit_col, quantity_col = self.find_financial_columns()

        if not revenue_col:
            print("No revenue column found for customer behavior analysis")
            return

        # Calculate RFM metrics
        try:
            current_date = self.merged_data['transaction_date'].max()

            rfm = self.merged_data.groupby('customer_id').agg({
                'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
                'transaction_id': 'count',  # Frequency
                revenue_col: 'sum'  # Monetary
            }).reset_index()

            rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

            plt.figure(figsize=(15, 5))

            # Recency distribution
            plt.subplot(1, 3, 1)
            sns.histplot(rfm['recency'], bins=20, kde=True)
            plt.title('Customer Recency Distribution')
            plt.xlabel('Days Since Last Purchase')

            # Frequency distribution
            plt.subplot(1, 3, 2)
            sns.histplot(rfm['frequency'], bins=20, kde=True)
            plt.title('Customer Frequency Distribution')
            plt.xlabel('Number of Purchases')

            # Monetary distribution
            plt.subplot(1, 3, 3)
            sns.histplot(rfm['monetary'], bins=20, kde=True)
            plt.title('Customer Monetary Value Distribution')
            plt.xlabel('Total Spending ($)')

            plt.tight_layout()
            plt.savefig('/content/customer_behavior_rfm.png', dpi=300, bbox_inches='tight')
            plt.show()

            # Correlation analysis
            plt.figure(figsize=(10, 8))

            # Select numerical columns for correlation
            numerical_cols = ['recency', 'frequency', 'monetary']
            correlation_data = rfm.copy()

            # Add age if available
            if 'age' in self.customers.columns and not self.customers.empty:
                correlation_data = correlation_data.merge(
                    self.customers[['customer_id', 'age']], on='customer_id', how='left'
                )
                numerical_cols.append('age')

            corr_matrix = correlation_data[numerical_cols].corr()

            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Correlation Matrix of Customer Metrics')
            plt.tight_layout()
            plt.savefig('/content/correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()

            # Print behavioral insights
            print(f"Average customer lifetime value: ${rfm['monetary'].mean():.2f}")
            print(f"Average purchase frequency: {rfm['frequency'].mean():.1f}")
            print(f"Average recency: {rfm['recency'].mean():.1f} days")

        except Exception as e:
            print(f"Error in customer behavior analysis: {e}")

    def product_analysis(self):
        """Analyze product performance"""
        if self.merged_data.empty:
            print("No merged data available for product analysis")
            return

        print("\n" + "=" * 50)
        print("PRODUCT PERFORMANCE ANALYSIS")
        print("=" * 50)

        # Find financial columns
        revenue_col, profit_col, quantity_col = self.find_financial_columns()

        if not revenue_col:
            print("No revenue column found for product analysis")
            return

        try:
            # Product performance metrics
            groupby_cols = ['product_name'] if 'product_name' in self.merged_data.columns else ['product_id']
            if 'category' in self.merged_data.columns:
                groupby_cols.append('category')

            product_performance = self.merged_data.groupby(groupby_cols).agg({
                revenue_col: 'sum',
                'transaction_id': 'count'
            }).reset_index()

            if profit_col:
                profit_data = self.merged_data.groupby(groupby_cols)[profit_col].sum().reset_index()
                product_performance = product_performance.merge(profit_data, on=groupby_cols)
                product_performance.columns = groupby_cols + ['total_revenue', 'transaction_count', 'total_profit']
            else:
                product_performance.columns = groupby_cols + ['total_revenue', 'transaction_count']

            if quantity_col and quantity_col in self.merged_data.columns:
                quantity_data = self.merged_data.groupby(groupby_cols)[quantity_col].sum().reset_index()
                product_performance = product_performance.merge(quantity_data, on=groupby_cols)
                product_performance.columns = list(product_performance.columns[:-1]) + ['total_quantity']

            # Top 10 products by revenue
            top_products = product_performance.nlargest(10, 'total_revenue')

            plt.figure(figsize=(15, 10))

            # Top products by revenue
            plt.subplot(2, 2, 1)
            sns.barplot(data=top_products, x='total_revenue', y=groupby_cols[0])
            plt.title('Top 10 Products by Revenue')
            plt.xlabel('Total Revenue ($)')

            # Price distribution from products table
            if not self.products.empty and 'price' in self.products.columns:
                plt.subplot(2, 2, 2)
                sns.histplot(self.products['price'], bins=20, kde=True)
                plt.title('Product Price Distribution')
                plt.xlabel('Price ($)')

            # Revenue vs Profit scatter plot (if profit available)
            if 'total_profit' in product_performance.columns:
                plt.subplot(2, 2, 3)
                plt.scatter(product_performance['total_revenue'],
                           product_performance['total_profit'], alpha=0.6)
                plt.xlabel('Total Revenue ($)')
                plt.ylabel('Total Profit ($)')
                plt.title('Revenue vs Profit by Product')
                plt.grid(True, alpha=0.3)

                # Add trend line
                z = np.polyfit(product_performance['total_revenue'],
                              product_performance['total_profit'], 1)
                p = np.poly1d(z)
                plt.plot(product_performance['total_revenue'],
                        p(product_performance['total_revenue']), "r--", alpha=0.8)

            plt.tight_layout()
            plt.savefig('/content/product_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

            # Print product insights
            print(f"Top selling product: {top_products.iloc[0][groupby_cols[0]]}")
            print(f"Average product revenue: ${product_performance['total_revenue'].mean():.2f}")
            if not self.products.empty and 'price' in self.products.columns:
                print(f"Average product price: ${self.products['price'].mean():.2f}")

        except Exception as e:
            print(f"Error in product analysis: {e}")

    def website_interaction_analysis(self):
        """Analyze website interaction patterns"""
        if self.interactions.empty:
            print("No interaction data available for website analysis")
            return

        print("\n" + "=" * 50)
        print("WEBSITE INTERACTION ANALYSIS")
        print("=" * 50)

        try:
            # Page visit distribution
            plt.figure(figsize=(15, 10))

            if 'page_visited' in self.interactions.columns:
                plt.subplot(2, 2, 1)
                page_visits = self.interactions['page_visited'].value_counts()
                sns.barplot(x=page_visits.values, y=page_visits.index)
                plt.title('Page Visits Distribution')
                plt.xlabel('Number of Visits')

            # Action distribution
            if 'action' in self.interactions.columns:
                plt.subplot(2, 2, 2)
                actions = self.interactions['action'].value_counts()
                sns.barplot(x=actions.values, y=actions.index)
                plt.title('User Actions Distribution')
                plt.xlabel('Number of Actions')

            # Session duration distribution
            if 'session_duration' in self.interactions.columns:
                plt.subplot(2, 2, 3)
                sns.histplot(self.interactions['session_duration'], bins=50, kde=True)
                plt.title('Session Duration Distribution')
                plt.xlabel('Session Duration (seconds)')

            # Products viewed distribution
            if 'products_viewed' in self.interactions.columns:
                plt.subplot(2, 2, 4)
                sns.histplot(self.interactions['products_viewed'], bins=20, kde=True)
                plt.title('Products Viewed Distribution')
                plt.xlabel('Number of Products Viewed')

            plt.tight_layout()
            plt.savefig('/content/website_interactions.png', dpi=300, bbox_inches='tight')
            plt.show()

            # Print interaction insights
            if 'page_visited' in self.interactions.columns:
                print(f"Most visited page: {page_visits.index[0]}")
            if 'action' in self.interactions.columns:
                print(f"Most common action: {actions.index[0]}")
            if 'session_duration' in self.interactions.columns:
                print(f"Average session duration: {self.interactions['session_duration'].mean():.1f} seconds")
            if 'products_viewed' in self.interactions.columns:
                print(f"Average products viewed per session: {self.interactions['products_viewed'].mean():.1f}")

        except Exception as e:
            print(f"Error in website interaction analysis: {e}")

    def generate_interactive_dashboard(self):
        """Create interactive dashboard using Plotly"""
        if self.merged_data.empty:
            print("No merged data available for interactive dashboard")
            return

        print("\nGenerating interactive dashboard...")

        try:
            # Find financial columns
            revenue_col, profit_col, quantity_col = self.find_financial_columns()

            if not revenue_col:
                print("No revenue column found for dashboard")
                return

            # Monthly sales trend
            monthly_sales = self.merged_data.groupby(
                pd.Grouper(key='transaction_date', freq='M')
            ).agg({
                revenue_col: 'sum',
                'transaction_id': 'count'
            }).reset_index()

            if profit_col:
                monthly_profit = self.merged_data.groupby(
                    pd.Grouper(key='transaction_date', freq='M')
                )[profit_col].sum().reset_index()
                monthly_sales = monthly_sales.merge(monthly_profit, on='transaction_date')

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Revenue Trend', 'Monthly Profit Trend' if profit_col else 'Transaction Count',
                              'Transaction Count Trend', 'Revenue by Category' if 'category' in self.merged_data.columns else 'Revenue Summary'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # Revenue trend
            fig.add_trace(
                go.Scatter(x=monthly_sales['transaction_date'],
                          y=monthly_sales[revenue_col],
                          mode='lines+markers',
                          name='Revenue',
                          line=dict(color='blue', width=3)),
                row=1, col=1
            )

            # Profit trend (if available)
            if profit_col and profit_col in monthly_sales.columns:
                fig.add_trace(
                    go.Scatter(x=monthly_sales['transaction_date'],
                              y=monthly_sales[profit_col],
                              mode='lines+markers',
                              name='Profit',
                              line=dict(color='green', width=3)),
                    row=1, col=2
                )
            else:
                # Show transaction count instead
                fig.add_trace(
                    go.Scatter(x=monthly_sales['transaction_date'],
                              y=monthly_sales['transaction_id'],
                              mode='lines+markers',
                              name='Transactions',
                              line=dict(color='orange', width=3)),
                    row=1, col=2
                )

            # Transaction count
            fig.add_trace(
                go.Scatter(x=monthly_sales['transaction_date'],
                          y=monthly_sales['transaction_id'],
                          mode='lines+markers',
                          name='Transactions',
                          line=dict(color='orange', width=3)),
                row=2, col=1
            )

            # Category revenue (if available)
            if 'category' in self.merged_data.columns:
                category_revenue = self.merged_data.groupby('category')[revenue_col].sum().sort_values(ascending=False)
                fig.add_trace(
                    go.Bar(x=category_revenue.values,
                          y=category_revenue.index,
                          orientation='h',
                          name='Category Revenue',
                          marker_color='lightblue'),
                    row=2, col=2
                )

            fig.update_layout(height=800, title_text="E-commerce Performance Dashboard")
            fig.write_html('/content/interactive_dashboard.html')

            print("Interactive dashboard saved as '/content/interactive_dashboard.html'")

        except Exception as e:
            print(f"Error generating interactive dashboard: {e}")

    def run_complete_analysis(self):
        """Execute complete EDA pipeline"""
        print("Starting Exploratory Data Analysis...")

        self.load_cleaned_data()
        self.basic_statistics()

        # Only run analyses if data is available
        if not self.customers.empty:
            self.customer_demographics_analysis()

        if not self.merged_data.empty:
            self.sales_analysis()
            self.customer_behavior_analysis()
            self.product_analysis()
            self.generate_interactive_dashboard()

        if not self.interactions.empty:
            self.website_interaction_analysis()

        print("\n" + "=" * 50)
        print("EXPLORATORY DATA ANALYSIS COMPLETED!")
        print("=" * 50)

if __name__ == "__main__":
    eda = EDA()
    eda.run_complete_analysis()