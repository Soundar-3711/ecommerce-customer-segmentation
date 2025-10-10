#6. Modeling and Customer Segmentation
# src/modeling.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """Perform customer segmentation using clustering algorithms"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def load_engineered_data(self):
        """Load the engineered feature dataset"""
        try:
            self.features_df = pd.read_csv('/content/engineered_features.csv')

            # Debug: Print available columns
            print("Available columns in engineered features:")
            print(list(self.features_df.columns))

            # Prepare features for clustering
            self.customer_ids = self.features_df['customer_id']

            # Get RFM segments if available
            if 'rfm_segment' in self.features_df.columns:
                self.rfm_segments = self.features_df['rfm_segment']
            else:
                self.rfm_segments = None
                print("Warning: RFM segments not found in data")

            # Select numerical features for clustering
            numerical_features = self.features_df.select_dtypes(include=[np.number]).columns

            # Remove RFM score columns if they exist
            columns_to_remove = ['recency_score', 'frequency_score', 'monetary_score', 'rfm_score']
            numerical_features = [col for col in numerical_features if col not in columns_to_remove]

            self.X = self.features_df[numerical_features]

            # Handle any remaining missing values
            self.X = self.X.fillna(0)

            print(f"Data loaded: {self.X.shape[0]} customers, {self.X.shape[1]} features")
            print(f"Features used: {list(self.X.columns)}")

        except Exception as e:
            print(f"Error loading engineered data: {e}")
            self.features_df = pd.DataFrame()
            self.X = pd.DataFrame()
            self.customer_ids = pd.Series()
            self.rfm_segments = None

    def get_available_columns_for_analysis(self):
        """Get available columns for cluster analysis with fallbacks"""
        available_columns = {}

        # Basic RFM columns (should be available if RFM was created)
        available_columns['recency'] = 'recency' if 'recency' in self.X.columns else None
        available_columns['frequency'] = 'frequency' if 'frequency' in self.X.columns else None
        available_columns['monetary'] = 'monetary' if 'monetary' in self.X.columns else None

        # Demographic columns
        available_columns['age'] = 'age' if 'age' in self.X.columns else None
        age_related_cols = [col for col in self.X.columns if 'age' in col.lower()]
        if not available_columns['age'] and age_related_cols:
            available_columns['age'] = age_related_cols[0]

        # Behavioral columns
        available_columns['total_interactions'] = 'total_interactions' if 'total_interactions' in self.X.columns else None
        interaction_cols = [col for col in self.X.columns if 'interaction' in col.lower() or 'session' in col.lower()]
        if not available_columns['total_interactions'] and interaction_cols:
            available_columns['total_interactions'] = interaction_cols[0]

        # Print available columns for debugging
        print("Available columns for cluster analysis:")
        for key, value in available_columns.items():
            print(f"  {key}: {value}")

        return available_columns

    def determine_optimal_clusters(self, max_clusters=10):
        """Determine optimal number of clusters using elbow method and silhouette analysis"""
        print("Determining optimal number of clusters...")

        if self.X.empty:
            print("No data available for clustering")
            return 3

        wcss = []  # Within-cluster sum of squares
        silhouette_scores = []

        for k in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(self.X)
                wcss.append(kmeans.inertia_)

                if k > 1:  # Silhouette score requires at least 2 clusters
                    silhouette_scores.append(silhouette_score(self.X, kmeans.labels_))
            except Exception as e:
                print(f"Error with k={k}: {e}")
                break

        if not wcss:
            print("Could not determine optimal clusters, using default k=3")
            return 3

        # Plot elbow curve and silhouette scores
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(2, len(wcss) + 2), wcss, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        plt.grid(True, alpha=0.3)

        if silhouette_scores:
            plt.subplot(1, 2, 2)
            plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o', color='green')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Analysis')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/content/optimal_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Find optimal k (using silhouette score if available)
        if silhouette_scores:
            optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range starts from 2
        else:
            # Use elbow method (find the "elbow" point)
            differences = np.diff(wcss)
            optimal_k = np.argmin(differences) + 2 if len(differences) > 0 else 3

        print(f"Optimal number of clusters: {optimal_k}")

        return optimal_k

    def perform_kmeans_clustering(self, n_clusters=4):
        """Perform K-means clustering"""
        print(f"Performing K-means clustering with {n_clusters} clusters...")

        if self.X.empty:
            print("No data available for clustering")
            return None

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X)

            # Calculate metrics
            silhouette_avg = silhouette_score(self.X, cluster_labels)
            calinski_score = calinski_harabasz_score(self.X, cluster_labels)
            davies_score = davies_bouldin_score(self.X, cluster_labels)

            print(f"K-means Results:")
            print(f"  Silhouette Score: {silhouette_avg:.3f}")
            print(f"  Calinski-Harabasz Score: {calinski_score:.3f}")
            print(f"  Davies-Bouldin Score: {davies_score:.3f}")

            self.models['kmeans'] = kmeans
            self.results['kmeans'] = {
                'labels': cluster_labels,
                'metrics': {
                    'silhouette': silhouette_avg,
                    'calinski_harabasz': calinski_score,
                    'davies_bouldin': davies_score
                }
            }

            return cluster_labels

        except Exception as e:
            print(f"Error in K-means clustering: {e}")
            return None

    def perform_gmm_clustering(self, n_clusters=4):
        """Perform Gaussian Mixture Model clustering"""
        print(f"Performing GMM clustering with {n_clusters} components...")

        if self.X.empty:
            print("No data available for clustering")
            return None

        try:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            cluster_labels = gmm.fit_predict(self.X)

            # Calculate metrics
            silhouette_avg = silhouette_score(self.X, cluster_labels)
            calinski_score = calinski_harabasz_score(self.X, cluster_labels)

            print(f"GMM Results:")
            print(f"  Silhouette Score: {silhouette_avg:.3f}")
            print(f"  Calinski-Harabasz Score: {calinski_score:.3f}")

            self.models['gmm'] = gmm
            self.results['gmm'] = {
                'labels': cluster_labels,
                'metrics': {
                    'silhouette': silhouette_avg,
                    'calinski_harabasz': calinski_score
                }
            }

            return cluster_labels

        except Exception as e:
            print(f"Error in GMM clustering: {e}")
            return None

    def perform_dimensionality_reduction(self):
        """Perform PCA and t-SNE for visualization"""
        print("Performing dimensionality reduction...")

        if self.X.empty:
            print("No data available for dimensionality reduction")
            return None, None

        try:
            # PCA
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(self.X)

            # t-SNE (with smaller dataset if too large)
            if len(self.X) > 1000:
                print("Using subset for t-SNE (dataset too large)...")
                sample_indices = np.random.choice(len(self.X), 1000, replace=False)
                X_sample = self.X.iloc[sample_indices]
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                X_tsne_full = np.zeros((len(self.X), 2))
                X_tsne_full[sample_indices] = tsne.fit_transform(X_sample)
                # For non-sampled points, use nearest neighbor in t-SNE space
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(X_tsne_full[sample_indices])
                non_sample_indices = [i for i in range(len(self.X)) if i not in sample_indices]
                if non_sample_indices:
                    _, indices = nn.kneighbors(self.X.iloc[non_sample_indices])
                    X_tsne_full[non_sample_indices] = X_tsne_full[sample_indices][indices.flatten()]
                X_tsne = X_tsne_full
            else:
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                X_tsne = tsne.fit_transform(self.X)

            self.dim_reduction = {
                'pca': {'transformed': X_pca, 'explained_variance': pca.explained_variance_ratio_},
                'tsne': {'transformed': X_tsne}
            }

            print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

            return X_pca, X_tsne

        except Exception as e:
            print(f"Error in dimensionality reduction: {e}")
            return None, None

    def visualize_clusters(self, cluster_labels, algorithm_name):
        """Visualize clustering results"""
        if cluster_labels is None:
            print(f"No cluster labels available for {algorithm_name}")
            return

        print(f"Visualizing {algorithm_name} clusters...")

        if not hasattr(self, 'dim_reduction') or self.dim_reduction is None:
            print("Performing dimensionality reduction first...")
            self.perform_dimensionality_reduction()

        if not hasattr(self, 'dim_reduction') or self.dim_reduction is None:
            print("Dimensionality reduction failed, cannot visualize clusters")
            return

        X_pca = self.dim_reduction['pca']['transformed']
        X_tsne = self.dim_reduction['tsne']['transformed']

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # PCA visualization
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                                 cmap='viridis', alpha=0.7, s=50)
        axes[0].set_title(f'{algorithm_name} Clusters - PCA')
        if 'pca' in self.dim_reduction and 'explained_variance' in self.dim_reduction['pca']:
            axes[0].set_xlabel(f'PC1 ({self.dim_reduction["pca"]["explained_variance"][0]:.2%} variance)')
            axes[0].set_ylabel(f'PC2 ({self.dim_reduction["pca"]["explained_variance"][1]:.2%} variance)')
        plt.colorbar(scatter1, ax=axes[0])

        # t-SNE visualization
        scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels,
                                 cmap='viridis', alpha=0.7, s=50)
        axes[1].set_title(f'{algorithm_name} Clusters - t-SNE')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=axes[1])

        plt.tight_layout()
        plt.savefig(f'/content/{algorithm_name.lower()}_clusters.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_cluster_characteristics(self, cluster_labels, algorithm_name):
        """Analyze and describe each cluster"""
        if cluster_labels is None:
            print(f"No cluster labels available for {algorithm_name} analysis")
            return None

        print(f"Analyzing {algorithm_name} cluster characteristics...")

        # Add cluster labels to features
        cluster_analysis_df = self.X.copy()
        cluster_analysis_df['cluster'] = cluster_labels
        cluster_analysis_df['customer_id'] = self.customer_ids.values

        if self.rfm_segments is not None:
            cluster_analysis_df['rfm_segment'] = self.rfm_segments.values

        # Get available columns for analysis
        available_cols = self.get_available_columns_for_analysis()

        # Create aggregation dictionary with only available columns
        agg_dict = {}
        for col_name, col_value in available_cols.items():
            if col_value and col_value in cluster_analysis_df.columns:
                agg_dict[col_value] = ['mean', 'std']

        # Add count of customers
        agg_dict['customer_id'] = 'count'

        if not agg_dict:
            print("No suitable columns found for cluster analysis")
            return cluster_analysis_df

        # Calculate cluster statistics
        try:
            cluster_stats = cluster_analysis_df.groupby('cluster').agg(agg_dict).round(2)

            # Flatten column names
            cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]

            print(f"\n{algorithm_name} Cluster Statistics:")
            print(cluster_stats)

        except Exception as e:
            print(f"Error calculating cluster statistics: {e}")
            # Fallback: simple count
            cluster_counts = cluster_analysis_df['cluster'].value_counts().sort_index()
            print(f"\nCluster sizes:")
            for cluster, count in cluster_counts.items():
                print(f"  Cluster {cluster}: {count} customers ({count/len(cluster_analysis_df)*100:.1f}%)")

        # Count customers per cluster
        cluster_counts = cluster_analysis_df['cluster'].value_counts().sort_index()
        print(f"\nCluster sizes:")
        for cluster, count in cluster_counts.items():
            print(f"  Cluster {cluster}: {count} customers ({count/len(cluster_analysis_df)*100:.1f}%)")

        # RFM segment distribution per cluster (if available)
        if 'rfm_segment' in cluster_analysis_df.columns:
            rfm_distribution = pd.crosstab(
                cluster_analysis_df['cluster'],
                cluster_analysis_df['rfm_segment'],
                normalize='index'
            ).round(3)

            print(f"\nRFM Segment Distribution per Cluster:")
            print(rfm_distribution)

        # Save cluster assignments
        columns_to_save = ['customer_id', 'cluster']
        if 'rfm_segment' in cluster_analysis_df.columns:
            columns_to_save.append('rfm_segment')

        cluster_assignments = cluster_analysis_df[columns_to_save]
        cluster_assignments['algorithm'] = algorithm_name
        cluster_assignments.to_csv(
            f'/content/{algorithm_name.lower()}_cluster_assignments.csv',
            index=False
        )

        return cluster_analysis_df

    def compare_clustering_algorithms(self):
        """Compare performance of different clustering algorithms"""
        print("Comparing clustering algorithms...")

        algorithms = ['kmeans', 'gmm']
        comparison_results = []

        for algo in algorithms:
            if algo in self.results and self.results[algo]['labels'] is not None:
                results = self.results[algo]
                comparison_results.append({
                    'Algorithm': algo.upper(),
                    'Silhouette Score': results['metrics']['silhouette'],
                    'Calinski-Harabasz Score': results['metrics'].get('calinski_harabasz', np.nan),
                    'Davies-Bouldin Score': results['metrics'].get('davies_bouldin', np.nan)
                })

        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            print("\nAlgorithm Comparison:")
            print(comparison_df.to_string(index=False))

            return comparison_df
        else:
            print("No clustering results available for comparison")
            return pd.DataFrame()

    def build_segment_classifier(self):
        """Build a classifier to predict customer segments"""
        print("Building segment classification model...")

        if self.X.empty or self.rfm_segments is None:
            print("No data or RFM segments available for classification")
            return None, None

        try:
            # Use RFM segments as target
            le = LabelEncoder()
            y = le.fit_transform(self.rfm_segments)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train Random Forest classifier
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier.fit(X_train, y_train)

            # Cross-validation scores
            cv_scores = cross_val_score(rf_classifier, self.X, y, cv=5)

            print(f"Segment Classification Results:")
            print(f"  Training Accuracy: {rf_classifier.score(X_train, y_train):.3f}")
            print(f"  Cross-validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': rf_classifier.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))

            # Plot feature importance
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(15)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('Top 15 Feature Importances for Segment Classification')
            plt.tight_layout()
            plt.savefig('/content/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()

            self.models['classifier'] = rf_classifier
            self.le = le

            return rf_classifier, feature_importance

        except Exception as e:
            print(f"Error building segment classifier: {e}")
            return None, None

    def generate_segment_profiles(self):
        """Generate detailed profiles for each customer segment"""
        print("Generating customer segment profiles...")

        # Use K-means results for profiling
        if 'kmeans' in self.results and self.results['kmeans']['labels'] is not None:
            cluster_labels = self.results['kmeans']['labels']
            cluster_analysis_df = self.analyze_cluster_characteristics(cluster_labels, 'KMeans')

            if cluster_analysis_df is None:
                print("No cluster analysis data available")
                return {}

            # Get available columns for profiling
            available_cols = self.get_available_columns_for_analysis()

            # Create detailed profiles
            segment_profiles = {}

            for cluster in sorted(cluster_analysis_df['cluster'].unique()):
                cluster_data = cluster_analysis_df[cluster_analysis_df['cluster'] == cluster]

                profile = {
                    'size': len(cluster_data),
                    'size_percentage': len(cluster_data) / len(cluster_analysis_df) * 100
                }

                # Add available metrics to profile
                if available_cols['recency']:
                    profile['avg_recency'] = cluster_data[available_cols['recency']].mean()
                if available_cols['frequency']:
                    profile['avg_frequency'] = cluster_data[available_cols['frequency']].mean()
                if available_cols['monetary']:
                    profile['avg_monetary'] = cluster_data[available_cols['monetary']].mean()
                if available_cols['age']:
                    profile['avg_age'] = cluster_data[available_cols['age']].mean()
                if available_cols['total_interactions']:
                    profile['avg_interactions'] = cluster_data[available_cols['total_interactions']].mean()

                # Add dominant RFM segment if available
                if 'rfm_segment' in cluster_data.columns:
                    profile['dominant_rfm_segment'] = cluster_data['rfm_segment'].mode().iloc[0] if not cluster_data['rfm_segment'].mode().empty else 'Unknown'

                segment_profiles[cluster] = profile

            # Print segment profiles
            print("\n" + "="*60)
            print("CUSTOMER SEGMENT PROFILES")
            print("="*60)

            for cluster, profile in segment_profiles.items():
                print(f"\n--- Segment {cluster} ---")
                print(f"Size: {profile['size']} customers ({profile['size_percentage']:.1f}%)")

                if 'avg_recency' in profile:
                    print(f"Average Recency: {profile['avg_recency']:.1f} days")
                if 'avg_frequency' in profile:
                    print(f"Average Frequency: {profile['avg_frequency']:.1f} purchases")
                if 'avg_monetary' in profile:
                    print(f"Average Monetary: ${profile['avg_monetary']:.2f}")
                if 'avg_age' in profile:
                    print(f"Average Age: {profile['avg_age']:.1f} years")
                if 'avg_interactions' in profile:
                    print(f"Average Interactions: {profile['avg_interactions']:.1f}")
                if 'dominant_rfm_segment' in profile:
                    print(f"Dominant RFM Segment: {profile['dominant_rfm_segment']}")

                # Segment interpretation based on available metrics
                self._interpret_segment(profile, cluster)

            return segment_profiles
        else:
            print("No K-means results available for segment profiling")
            return {}

    def _interpret_segment(self, profile, cluster):
        """Interpret segment based on available metrics"""
        interpretation = "REGULAR CUSTOMERS"
        recommendation = "Standard marketing, loyalty programs"

        # Check if we have enough metrics for interpretation
        has_recency = 'avg_recency' in profile
        has_frequency = 'avg_frequency' in profile
        has_monetary = 'avg_monetary' in profile

        if has_recency and has_frequency and has_monetary:
            if (profile['avg_recency'] < 30 and
                profile['avg_frequency'] > 5 and
                profile['avg_monetary'] > 1000):
                interpretation = "HIGH-VALUE LOYAL CUSTOMERS"
                recommendation = "Premium rewards, exclusive offers, VIP treatment"
            elif (profile['avg_recency'] > 90 and
                  profile['avg_frequency'] < 2):
                interpretation = "AT-RISK CUSTOMERS"
                recommendation = "Win-back campaigns, special discounts"
            elif (profile['avg_recency'] < 60 and
                  profile['avg_monetary'] < 200):
                interpretation = "NEW/ACTIVE LOW-SPENDERS"
                recommendation = "Upselling, product recommendations"

        print(f"Interpretation: {interpretation}")
        print(f"Recommendation: {recommendation}")

    def run_complete_modeling_pipeline(self):
        """Execute complete modeling pipeline"""
        print("Starting Customer Segmentation Modeling Pipeline...")

        self.load_engineered_data()

        if self.X.empty:
            print("No data available for modeling pipeline")
            return {}

        # Determine optimal clusters
        optimal_k = self.determine_optimal_clusters()

        # Perform clustering
        kmeans_labels = self.perform_kmeans_clustering(optimal_k)
        gmm_labels = self.perform_gmm_clustering(optimal_k)

        # Dimensionality reduction for visualization
        self.perform_dimensionality_reduction()

        # Visualize results
        if kmeans_labels is not None:
            self.visualize_clusters(kmeans_labels, 'KMeans')
        if gmm_labels is not None:
            self.visualize_clusters(gmm_labels, 'GMM')

        # Analyze clusters
        if kmeans_labels is not None:
            self.analyze_cluster_characteristics(kmeans_labels, 'KMeans')
        if gmm_labels is not None:
            self.analyze_cluster_characteristics(gmm_labels, 'GMM')

        # Compare algorithms
        self.compare_clustering_algorithms()

        # Build classifier (if RFM segments available)
        if self.rfm_segments is not None:
            self.build_segment_classifier()

        # Generate segment profiles
        segment_profiles = self.generate_segment_profiles()

        print("\n" + "="*50)
        print("MODELING PIPELINE COMPLETED!")
        print("="*50)

        return segment_profiles

if __name__ == "__main__":
    segmentation = CustomerSegmentation()
    segment_profiles = segmentation.run_complete_modeling_pipeline()