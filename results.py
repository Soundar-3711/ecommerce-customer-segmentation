#7. Results and Business Insights
# src/results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BusinessInsights:
    """Generate business insights and recommendations from analysis"""

    def __init__(self):
        self.insights = {}

    def load_all_data(self):
        """Load all processed data and results"""
        try:
            # Load main datasets
            self.customers = pd.read_csv('/content/customers_cleaned.csv')
            self.products = pd.read_csv('/content/products_cleaned.csv')
            self.transactions = pd.read_csv('/content/transactions_cleaned.csv')
            self.merged_data = pd.read_csv('/content/merged_data.csv')
            self.engineered_features = pd.read_csv('/content/engineered_features.csv')

            # Load clustering results if available
            try:
                self.kmeans_clusters = pd.read_csv('/content/kmeans_cluster_assignments.csv')
            except FileNotFoundError:
                print("K-means cluster assignments not found, creating empty DataFrame")
                self.kmeans_clusters = pd.DataFrame()

            # Convert date columns
            date_columns = {
                'customers': ['signup_date'],
                'transactions': ['transaction_date'],
                'merged_data': ['transaction_date']
            }

            for df_name, cols in date_columns.items():
                df = getattr(self, df_name)
                for col in cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])

            print("All data loaded successfully!")

            # Debug: Print available columns
            print("\nAvailable columns in key datasets:")
            print("Merged data:", list(self.merged_data.columns) if not self.merged_data.empty else "Empty")
            print("Engineered features:", list(self.engineered_features.columns) if not self.engineered_features.empty else "Empty")

        except Exception as e:
            print(f"Error loading data: {e}")
            # Create empty dataframes
            self.customers = pd.DataFrame()
            self.products = pd.DataFrame()
            self.transactions = pd.DataFrame()
            self.merged_data = pd.DataFrame()
            self.engineered_features = pd.DataFrame()
            self.kmeans_clusters = pd.DataFrame()

    def find_financial_columns(self):
        """Find available financial columns in the dataset"""
        possible_revenue_cols = ['revenue', 'amount', 'price', 'total_amount', 'sales', 'transaction_amount']
        possible_profit_cols = ['profit', 'profit_amount', 'margin']
        possible_quantity_cols = ['quantity', 'qty', 'units']

        # Find revenue column
        self.revenue_col = None
        if not self.transactions.empty:
            for col in possible_revenue_cols:
                if col in self.transactions.columns:
                    self.revenue_col = col
                    break

        # Find profit column
        self.profit_col = None
        if not self.transactions.empty:
            for col in possible_profit_cols:
                if col in self.transactions.columns:
                    self.profit_col = col
                    break

        # Find quantity column
        self.quantity_col = None
        if not self.transactions.empty:
            for col in possible_quantity_cols:
                if col in self.transactions.columns:
                    self.quantity_col = col
                    break

        print(f"Financial columns found - Revenue: {self.revenue_col}, Profit: {self.profit_col}, Quantity: {self.quantity_col}")

        return self.revenue_col, self.profit_col, self.quantity_col

    def get_product_identifier(self):
        """Find available product identifier column"""
        possible_product_cols = ['product_name', 'product_id', 'product', 'item_name']

        self.product_col = None
        if not self.merged_data.empty:
            for col in possible_product_cols:
                if col in self.merged_data.columns:
                    self.product_col = col
                    break

        print(f"Product identifier column: {self.product_col}")
        return self.product_col

    def get_category_column(self):
        """Find available category column"""
        possible_category_cols = ['category', 'product_category', 'category_name', 'type']

        self.category_col = None
        if not self.merged_data.empty:
            for col in possible_category_cols:
                if col in self.merged_data.columns:
                    self.category_col = col
                    break

        print(f"Category column: {self.category_col}")
        return self.category_col

    def generate_executive_summary(self):
        """Generate executive summary of key findings"""
        print("GENERATING EXECUTIVE SUMMARY")
        print("="*50)

        # Find financial columns
        revenue_col, profit_col, quantity_col = self.find_financial_columns()

        if not revenue_col:
            print("No revenue data available for executive summary")
            return

        # Key metrics
        total_revenue = self.transactions[revenue_col].sum()
        total_customers = self.customers['customer_id'].nunique() if not self.customers.empty else 0
        total_products = self.products['product_id'].nunique() if not self.products.empty else 0
        avg_transaction_value = self.transactions[revenue_col].mean()

        # Customer segments from clustering
        if not self.kmeans_clusters.empty and 'cluster' in self.kmeans_clusters.columns:
            segment_distribution = self.kmeans_clusters['cluster'].value_counts().sort_index()
        else:
            segment_distribution = pd.Series()

        # Top performing products
        product_col = self.get_product_identifier()
        if product_col and not self.merged_data.empty:
            top_products = self.merged_data.groupby(product_col)[revenue_col].sum().nlargest(5)
        else:
            top_products = pd.Series()

        print(f"\nKEY BUSINESS METRICS:")
        print(f"  Total Revenue: ${total_revenue:,.2f}")
        print(f"  Total Customers: {total_customers:,}")
        print(f"  Total Products: {total_products:,}")
        print(f"  Average Transaction Value: ${avg_transaction_value:.2f}")

        if not segment_distribution.empty:
            print(f"\nCUSTOMER SEGMENT DISTRIBUTION:")
            for segment, count in segment_distribution.items():
                percentage = (count / len(self.kmeans_clusters)) * 100
                print(f"  Segment {segment}: {count} customers ({percentage:.1f}%)")

        if not top_products.empty:
            print(f"\nTOP 5 PRODUCTS BY REVENUE:")
            for product, revenue in top_products.items():
                print(f"  {product}: ${revenue:,.2f}")

        # Store insights
        self.insights['executive_summary'] = {
            'total_revenue': total_revenue,
            'total_customers': total_customers,
            'avg_transaction_value': avg_transaction_value,
            'segment_distribution': segment_distribution.to_dict() if not segment_distribution.empty else {},
            'top_products': top_products.to_dict() if not top_products.empty else {}
        }

    def analyze_customer_lifetime_value(self):
        """Analyze customer lifetime value patterns"""
        print("\nCUSTOMER LIFETIME VALUE ANALYSIS")
        print("="*50)

        if self.engineered_features.empty:
            print("No engineered features available for CLV analysis")
            return

        # Check for required columns
        required_cols = ['monetary', 'rfm_segment']
        missing_cols = [col for col in required_cols if col not in self.engineered_features.columns]

        if missing_cols:
            print(f"Missing columns for CLV analysis: {missing_cols}")
            return

        try:
            # Calculate CLV by segment
            clv_by_segment = self.engineered_features.groupby('rfm_segment').agg({
                'monetary': ['mean', 'sum', 'count']
            }).round(2)

            clv_by_segment.columns = ['avg_clv', 'total_value', 'customer_count']
            clv_by_segment['percentage_of_value'] = (
                clv_by_segment['total_value'] / clv_by_segment['total_value'].sum() * 100
            ).round(1)

            print("Customer Lifetime Value by RFM Segment:")
            print(clv_by_segment)

            # CLV by cluster (if available)
            if not self.kmeans_clusters.empty and 'customer_id' in self.kmeans_clusters.columns:
                cluster_clv = self.kmeans_clusters.merge(
                    self.engineered_features[['customer_id', 'monetary']],
                    on='customer_id', how='left'
                )

                if 'cluster' in cluster_clv.columns:
                    cluster_clv_summary = cluster_clv.groupby('cluster').agg({
                        'monetary': ['mean', 'sum', 'count']
                    }).round(2)

                    cluster_clv_summary.columns = ['avg_clv', 'total_value', 'customer_count']

                    print("\nCustomer Lifetime Value by Cluster:")
                    print(cluster_clv_summary)
                else:
                    cluster_clv_summary = pd.DataFrame()
            else:
                cluster_clv_summary = pd.DataFrame()

            # Visualization
            plt.figure(figsize=(15, 6))

            plt.subplot(1, 2, 1)
            segments_plot = clv_by_segment.sort_values('avg_clv', ascending=False)
            sns.barplot(x=segments_plot['avg_clv'], y=segments_plot.index)
            plt.title('Average CLV by RFM Segment')
            plt.xlabel('Average Customer Lifetime Value ($)')

            if not cluster_clv_summary.empty:
                plt.subplot(1, 2, 2)
                clusters_plot = cluster_clv_summary.sort_values('avg_clv', ascending=False)
                sns.barplot(x=clusters_plot['avg_clv'], y=clusters_plot.index)
                plt.title('Average CLV by Cluster')
                plt.xlabel('Average Customer Lifetime Value ($)')

            plt.tight_layout()
            plt.savefig('/content/clv_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

            # Store insights
            self.insights['clv_analysis'] = {
                'rfm_segment_clv': clv_by_segment.to_dict(),
                'cluster_clv': cluster_clv_summary.to_dict() if not cluster_clv_summary.empty else {}
            }

        except Exception as e:
            print(f"Error in CLV analysis: {e}")

    def identify_growth_opportunities(self):
        """Identify potential growth opportunities"""
        print("\nGROWTH OPPORTUNITIES ANALYSIS")
        print("="*50)

        if self.merged_data.empty:
            print("No merged data available for growth opportunities analysis")
            return

        revenue_col, profit_col, quantity_col = self.find_financial_columns()
        product_col = self.get_product_identifier()

        if not revenue_col or not product_col:
            print("Required columns not available for growth opportunities analysis")
            return

        try:
            # Underperforming products
            agg_dict = {
                revenue_col: 'sum',
                'transaction_id': 'count'
            }

            if profit_col and profit_col in self.merged_data.columns:
                agg_dict[profit_col] = 'sum'
            if quantity_col and quantity_col in self.merged_data.columns:
                agg_dict[quantity_col] = 'sum'

            product_performance = self.merged_data.groupby(product_col).agg(agg_dict)

            # Calculate profit margin if profit data available
            if profit_col and profit_col in product_performance.columns:
                product_performance['profit_margin'] = (
                    product_performance[profit_col] / product_performance[revenue_col] * 100
                ).round(1)

                # Low revenue but high margin products (opportunity)
                opportunity_products = product_performance[
                    (product_performance[revenue_col] < product_performance[revenue_col].median()) &
                    (product_performance['profit_margin'] > product_performance['profit_margin'].median())
                ].sort_values('profit_margin', ascending=False)

                print("High-Margin, Low-Revenue Products (Growth Opportunities):")
                print(opportunity_products.head(10))
            else:
                opportunity_products = pd.DataFrame()
                print("No profit data available for margin analysis")

            # Customer acquisition priority based on RFM
            if not self.engineered_features.empty and 'monetary' in self.engineered_features.columns and 'frequency' in self.engineered_features.columns:
                segment_acquisition_priority = self.engineered_features.groupby('rfm_segment').agg({
                    'monetary': 'mean',
                    'frequency': 'mean'
                }).sort_values('monetary', ascending=False)

                print("\nCustomer Segments by Acquisition Priority:")
                print(segment_acquisition_priority)
            else:
                segment_acquisition_priority = pd.DataFrame()

            # Store insights
            self.insights['growth_opportunities'] = {
                'opportunity_products': opportunity_products.head(10).to_dict() if not opportunity_products.empty else {},
                'acquisition_priority': segment_acquisition_priority.to_dict() if not segment_acquisition_priority.empty else {}
            }

        except Exception as e:
            print(f"Error in growth opportunities analysis: {e}")

    def generate_marketing_recommendations(self):
        """Generate targeted marketing recommendations"""
        print("\nMARKETING RECOMMENDATIONS")
        print("="*50)

        # Segment-based recommendations (generic - will work even without cluster data)
        segment_strategies = {
            0: {
                'description': 'High-Value Loyal Customers',
                'characteristics': 'High spending, frequent purchases, recent activity',
                'recommendations': [
                    'Exclusive VIP program with early access to new products',
                    'Personalized product recommendations',
                    'Loyalty rewards and special discounts',
                    'Invite-only events and previews'
                ]
            },
            1: {
                'description': 'At-Risk High Spenders',
                'characteristics': 'High historical value but declining activity',
                'recommendations': [
                    'Win-back campaigns with special offers',
                    'Personalized reactivation emails',
                    'Survey to understand reasons for decreased activity',
                    'Limited-time exclusive discounts'
                ]
            },
            2: {
                'description': 'Active Low-Spenders',
                'characteristics': 'Regular activity but low transaction value',
                'recommendations': [
                    'Upselling and cross-selling campaigns',
                    'Product bundle offers',
                    'Educational content about premium features',
                    'Loyalty program introduction'
                ]
            },
            3: {
                'description': 'New/Low-Engagement Customers',
                'characteristics': 'Recent signups or infrequent activity',
                'recommendations': [
                    'Welcome series and onboarding campaigns',
                    'Introduction to key products and features',
                    'Small incentive for first major purchase',
                    'Engagement-focused content marketing'
                ]
            }
        }

        # Adjust strategies based on actual cluster count
        if not self.kmeans_clusters.empty and 'cluster' in self.kmeans_clusters.columns:
            actual_clusters = self.kmeans_clusters['cluster'].nunique()
            # Use only the strategies for clusters that exist
            segment_strategies = {k: v for k, v in segment_strategies.items() if k < actual_clusters}

        for segment, strategy in segment_strategies.items():
            print(f"\n--- Segment {segment}: {strategy['description']} ---")
            print(f"Characteristics: {strategy['characteristics']}")
            print("Recommended Actions:")
            for i, recommendation in enumerate(strategy['recommendations'], 1):
                print(f"  {i}. {recommendation}")

        # Product-specific recommendations
        category_col = self.get_category_column()
        revenue_col, _, _ = self.find_financial_columns()

        if category_col and revenue_col and not self.merged_data.empty:
            top_categories = self.merged_data.groupby(category_col)[revenue_col].sum().nlargest(3)
            print(f"\nTOP PERFORMING CATEGORIES:")
            for category, revenue in top_categories.items():
                print(f"  {category}: ${revenue:,.2f}")
                # Category-specific recommendations
                if 'electronics' in str(category).lower():
                    print("    → Focus on accessory bundles and extended warranties")
                elif 'computer' in str(category).lower():
                    print("    → Promote software bundles and tech support services")
                elif 'mobile' in str(category).lower():
                    print("    → Offer device protection plans and accessory packages")
                else:
                    print("    → Develop targeted promotions and bundle offers")

        # Store recommendations
        self.insights['marketing_recommendations'] = segment_strategies

    def generate_operational_insights(self):
        """Generate operational efficiency insights"""
        print("\nOPERATIONAL EFFICIENCY INSIGHTS")
        print("="*50)

        if self.merged_data.empty:
            print("No merged data available for operational insights")
            return

        revenue_col, _, quantity_col = self.find_financial_columns()
        product_col = self.get_product_identifier()

        if not revenue_col or not product_col:
            print("Required columns not available for operational insights")
            return

        try:
            # Inventory optimization analysis
            agg_dict = {
                revenue_col: 'sum'
            }

            if quantity_col and quantity_col in self.merged_data.columns:
                agg_dict[quantity_col] = 'sum'

            product_turnover = self.merged_data.groupby(product_col).agg(agg_dict).sort_values(revenue_col, ascending=False)

            # ABC analysis
            product_turnover['revenue_cumulative_percentage'] = (
                product_turnover[revenue_col].cumsum() / product_turnover[revenue_col].sum() * 100
            )

            # Classify products
            def abc_classification(row):
                if row['revenue_cumulative_percentage'] <= 80:
                    return 'A'
                elif row['revenue_cumulative_percentage'] <= 95:
                    return 'B'
                else:
                    return 'C'

            product_turnover['abc_class'] = product_turnover.apply(abc_classification, axis=1)

            abc_summary = product_turnover.groupby('abc_class').agg({
                product_col: 'count',
                revenue_col: 'sum'
            })

            abc_summary['revenue_percentage'] = (
                abc_summary[revenue_col] / abc_summary[revenue_col].sum() * 100
            ).round(1)

            print("ABC Analysis of Products:")
            print(abc_summary)

            # Seasonal patterns (if date column available)
            monthly_patterns = pd.DataFrame()
            if 'transaction_date' in self.merged_data.columns:
                monthly_patterns = self.merged_data.groupby(
                    self.merged_data['transaction_date'].dt.month
                ).agg({
                    revenue_col: 'sum',
                    'transaction_id': 'count'
                })

                print("\nMonthly Revenue Patterns:")
                print(monthly_patterns)

            # Store operational insights
            self.insights['operational_efficiency'] = {
                'abc_analysis': abc_summary.to_dict(),
                'monthly_patterns': monthly_patterns.to_dict() if not monthly_patterns.empty else {}
            }

        except Exception as e:
            print(f"Error in operational insights analysis: {e}")

    def create_final_report(self):
        """Create comprehensive final report"""
        print("\nGENERATING COMPREHENSIVE BUSINESS REPORT")
        print("="*50)

        # Generate all insights
        self.generate_executive_summary()
        self.analyze_customer_lifetime_value()
        self.identify_growth_opportunities()
        self.generate_marketing_recommendations()
        self.generate_operational_insights()

        # Create summary visualization if we have data
        if (not self.kmeans_clusters.empty and
            not self.engineered_features.empty and
            not self.merged_data.empty):

            try:
                plt.figure(figsize=(15, 10))

                # 1. Revenue by segment
                plt.subplot(2, 2, 1)
                if 'customer_id' in self.kmeans_clusters.columns and 'monetary' in self.engineered_features.columns:
                    segment_revenue = self.kmeans_clusters.merge(
                        self.engineered_features[['customer_id', 'monetary']],
                        on='customer_id'
                    ).groupby('cluster')['monetary'].sum()

                    plt.pie(segment_revenue.values, labels=segment_revenue.index, autopct='%1.1f%%')
                    plt.title('Revenue Distribution by Customer Segment')

                # 2. Customer distribution
                plt.subplot(2, 2, 2)
                if 'cluster' in self.kmeans_clusters.columns:
                    segment_counts = self.kmeans_clusters['cluster'].value_counts().sort_index()
                    sns.barplot(x=segment_counts.index, y=segment_counts.values)
                    plt.title('Customer Distribution by Segment')
                    plt.xlabel('Segment')
                    plt.ylabel('Number of Customers')

                # 3. Average CLV by segment
                plt.subplot(2, 2, 3)
                if ('customer_id' in self.kmeans_clusters.columns and
                    'monetary' in self.engineered_features.columns):
                    segment_clv = self.kmeans_clusters.merge(
                        self.engineered_features[['customer_id', 'monetary']],
                        on='customer_id'
                    ).groupby('cluster')['monetary'].mean()

                    sns.barplot(x=segment_clv.index, y=segment_clv.values)
                    plt.title('Average Customer Lifetime Value by Segment')
                    plt.xlabel('Segment')
                    plt.ylabel('Average CLV ($)')

                # 4. Monthly revenue trend
                plt.subplot(2, 2, 4)
                revenue_col, _, _ = self.find_financial_columns()
                if revenue_col and 'transaction_date' in self.merged_data.columns:
                    monthly_revenue = self.merged_data.groupby(
                        self.merged_data['transaction_date'].dt.month
                    )[revenue_col].sum()

                    plt.plot(monthly_revenue.index, monthly_revenue.values, marker='o')
                    plt.title('Monthly Revenue Trend')
                    plt.xlabel('Month')
                    plt.ylabel('Revenue ($)')
                    plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig('/content/business_insights_summary.png',
                           dpi=300, bbox_inches='tight')
                plt.show()

            except Exception as e:
                print(f"Error creating summary visualization: {e}")

        # Save insights to file
        try:
            import json
            with open('/content/business_insights.json', 'w') as f:
                json.dump(self.insights, f, indent=2, default=str)
            print("Business insights saved to '/content/business_insights.json'")
        except Exception as e:
            print(f"Error saving insights to file: {e}")

        print("\n" + "="*50)
        print("BUSINESS INSIGHTS REPORT COMPLETED!")
        print("="*50)
        print("\nKey Deliverables:")
        print("✓ Executive summary with key metrics")
        print("✓ Customer segmentation analysis")
        print("✓ Customer lifetime value assessment")
        print("✓ Growth opportunity identification")
        print("✓ Targeted marketing recommendations")
        print("✓ Operational efficiency insights")
        print("✓ Comprehensive visualizations")
        print("✓ JSON report with all insights")

if __name__ == "__main__":
    insights = BusinessInsights()
    insights.load_all_data()
    insights.create_final_report()