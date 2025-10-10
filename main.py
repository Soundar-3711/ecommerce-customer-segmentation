#8. Main Project Execution Script

# main.py
"""
E-commerce Customer Segmentation and Sales Analysis
Complete Data Science Project Pipeline
"""

import os
import sys
from datetime import datetime

def create_project_structure():
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'notebooks',
        'src',
        'models',
        'reports/figures'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("Project structure created successfully!")

def run_data_collection():
    """Run data collection phase"""
    print("\n" + "="*60)
    print("PHASE 1: DATA COLLECTION")
    print("="*60)
    
    from src.data_collection import collect_data
    
    customers, products, transactions, interactions = collect_data()
    
    print(f"✓ Generated {len(customers)} customer records")
    print(f"✓ Generated {len(products)} product records")
    print(f"✓ Generated {len(transactions)} transaction records")
    print(f"✓ Generated {len(interactions)} interaction records")
    
    return True

def run_data_cleaning():
    """Run data cleaning phase"""
    print("\n" + "="*60)
    print("PHASE 2: DATA CLEANING AND PREPROCESSING")
    print("="*60)
    
    from src.data_cleaning import DataCleaner
    
    cleaner = DataCleaner()
    cleaner.run_cleaning_pipeline()
    
    print("✓ Data cleaning completed")
    print("✓ Missing values handled")
    print("✓ Data validation performed")
    print("✓ Merged dataset created")
    
    return True

def run_exploratory_analysis():
    """Run exploratory data analysis"""
    print("\n" + "="*60)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    from src.eda import EDA
    
    eda = EDA()
    eda.run_complete_analysis()
    
    print("✓ Basic statistics generated")
    print("✓ Customer demographics analyzed")
    print("✓ Sales patterns identified")
    print("✓ Customer behavior analyzed")
    print("✓ Product performance assessed")
    print("✓ Interactive dashboard created")
    
    return True

def run_feature_engineering():
    """Run feature engineering phase"""
    print("\n" + "="*60)
    print("PHASE 4: FEATURE ENGINEERING")
    print("="*60)
    
    from src.feature_engineering import FeatureEngineer
    
    engineer = FeatureEngineer()
    modeling_data = engineer.run_feature_engineering_pipeline()
    
    print("✓ RFM features created")
    print("✓ Behavioral features engineered")
    print("✓ Demographic features processed")
    print("✓ Product preference features generated")
    print("✓ Comprehensive feature dataset prepared")
    
    return True

def run_modeling():
    """Run modeling and segmentation phase"""
    print("\n" + "="*60)
    print("PHASE 5: MODELING AND CUSTOMER SEGMENTATION")
    print("="*60)
    
    from src.modeling import CustomerSegmentation
    
    segmentation = CustomerSegmentation()
    segment_profiles = segmentation.run_complete_modeling_pipeline()
    
    print("✓ Optimal clusters determined")
    print("✓ Multiple clustering algorithms compared")
    print("✓ Customer segments identified")
    print("✓ Segment profiles generated")
    print("✓ Classification model built")
    
    return True

def generate_business_insights():
    """Generate final business insights and recommendations"""
    print("\n" + "="*60)
    print("PHASE 6: BUSINESS INSIGHTS AND RECOMMENDATIONS")
    print("="*60)
    
    from src.results import BusinessInsights
    
    insights = BusinessInsights()
    insights.load_all_data()
    insights.create_final_report()
    
    print("✓ Executive summary generated")
    print("✓ Customer lifetime value analyzed")
    print("✓ Growth opportunities identified")
    print("✓ Marketing recommendations provided")
    print("✓ Operational insights delivered")
    print("✓ Comprehensive report created")
    
    return True

def main():
    """Main function to run the complete project pipeline"""
    print("E-COMMERCE CUSTOMER SEGMENTATION AND SALES ANALYSIS")
    print("COMPLETE DATA SCIENCE PROJECT PIPELINE")
    print("="*60)
    
    start_time = datetime.now()
    print(f"Project started at: {start_time}")
    
    try:
        # Create project structure
        create_project_structure()
        
        # Run all phases
        phases = [
            ("Data Collection", run_data_collection),
            ("Data Cleaning", run_data_cleaning),
            ("Exploratory Analysis", run_exploratory_analysis),
            ("Feature Engineering", run_feature_engineering),
            ("Modeling", run_modeling),
            ("Business Insights", generate_business_insights)
        ]
        
        for phase_name, phase_function in phases:
            print(f"\n>>> Starting {phase_name.upper()}...")
            success = phase_function()
            if success:
                print(f">>> {phase_name.upper()} COMPLETED SUCCESSFULLY")
            else:
                print(f">>> {phase_name.upper()} FAILED")
                break
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("PROJECT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Start Time: {start_time}")
        print(f"End Time: {end_time}")
        print(f"Total Duration: {duration}")
        
        print("\nDELIVERABLES GENERATED:")
        print("✓ Raw and processed datasets")
        print("✓ Comprehensive EDA with visualizations")
        print("✓ Engineered features for modeling")
        print("✓ Customer segmentation models")
        print("✓ Business insights and recommendations")
        print("✓ Interactive dashboard")
        print("✓ Complete project documentation")
        
        print("\nNext steps:")
        print("1. Review reports/ directory for insights")
        print("2. Check reports/figures/ for visualizations")
        print("3. Examine data/processed/ for cleaned data")
        print("4. Review business_insights.json for recommendations")
        
    except Exception as e:
        print(f"\nERROR: Project failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)