#!/usr/bin/env python3
"""
Test script for ALS Clustering Analysis Pipeline

This script performs comprehensive testing of all components:
1. Dependencies and imports
2. Data availability and format
3. Individual module functionality
4. End-to-end pipeline execution

Author: ALS Research Team
Date: 2025
"""

import sys
import os
import time
from datetime import datetime
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_dependencies():
    """Test if all required dependencies are available."""
    print("🔍 Testing Dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 
        'scipy', 'kneed', 'nimfa'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ❌ {package} - {str(e)}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies are available!")
        return True


def test_imports():
    """Test if all custom modules can be imported."""
    print("\n🔍 Testing Module Imports...")
    
    modules_to_test = [
        'src.config',
        'src.data_loader', 
        'src.nmf_clustering',
        'src.visualization',
        'src.cross_modal_analysis',
        'src.results_export'
    ]
    
    import_errors = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ❌ {module} - {str(e)}")
            import_errors.append((module, str(e)))
    
    if import_errors:
        print(f"\n❌ Import errors found:")
        for module, error in import_errors:
            print(f"  {module}: {error}")
        return False
    else:
        print("✅ All modules imported successfully!")
        return True


def test_data_availability():
    """Test if required data files are available."""
    print("\n🔍 Testing Data Availability...")
    
    from src.config import ACOUSTIC_FEATURES_PATH, PERCEPTUAL_DATA_PATH, DEMOGRAPHICS_PATH
    import glob
    
    # Test acoustic features
    acoustic_files = glob.glob(ACOUSTIC_FEATURES_PATH)
    print(f"  Acoustic features: {len(acoustic_files)} files found")
    if len(acoustic_files) == 0:
        print(f"  ❌ No acoustic feature files found at: {ACOUSTIC_FEATURES_PATH}")
        return False
    else:
        print(f"  ✓ Found {len(acoustic_files)} acoustic feature files")
    
    # Test perceptual data
    if os.path.exists(PERCEPTUAL_DATA_PATH):
        print(f"  ✓ Perceptual data file found")
    else:
        print(f"  ❌ Perceptual data file not found: {PERCEPTUAL_DATA_PATH}")
        return False
    
    # Test demographics data
    if os.path.exists(DEMOGRAPHICS_PATH):
        print(f"  ✓ Demographics data file found")
    else:
        print(f"  ❌ Demographics data file not found: {DEMOGRAPHICS_PATH}")
        return False
    
    print("✅ All required data files are available!")
    return True


def test_data_loading():
    """Test data loading functionality."""
    print("\n🔍 Testing Data Loading...")
    
    try:
        from src.data_loader import merge_all_data, filter_demographics, get_demographic_stats
        
        print("  Loading and merging data...")
        all_data, clinical_summaries, demographics, first_acoustic = merge_all_data()
        
        print(f"  ✓ Loaded {all_data.shape[0]} participants with {all_data.shape[1]} features")
        print(f"  ✓ First acoustic feature at column {first_acoustic}")
        print(f"  ✓ Clinical summaries shape: {clinical_summaries.shape}")
        print(f"  ✓ Demographics shape: {demographics.shape}")
        
        # Test filtering
        demographics_filtered, clinical_dem_filtered = filter_demographics(demographics, clinical_summaries)
        print(f"  ✓ Filtered demographics: {demographics_filtered.shape[0]} participants")
        
        # Test demographic stats
        demographic_stats = get_demographic_stats(demographics_filtered)
        print(f"  ✓ Demographic statistics calculated for {len(demographic_stats)} variables")
        
        print("✅ Data loading test passed!")
        return True, (all_data, clinical_summaries, demographics, first_acoustic, demographic_stats)
        
    except Exception as e:
        print(f"  ❌ Data loading failed: {str(e)}")
        traceback.print_exc()
        return False, None


def test_nmf_clustering(test_data):
    """Test NMF clustering functionality."""
    print("\n🔍 Testing NMF Clustering...")
    
    try:
        from src.nmf_clustering import run_nmf_clustering, get_top_features
        
        all_data, clinical_summaries, demographics, first_acoustic, demographic_stats = test_data
        
        print("  Running NMF clustering (this may take a few minutes)...")
        all_clusters, all_feature_maps, coef_dists, raw_coef_dists, basis_dists = run_nmf_clustering(
            all_data, clinical_summaries, first_acoustic
        )
        
        print(f"  ✓ Acoustic clusters: {len(all_clusters['acoustic'][0])} assignments")
        print(f"  ✓ Perceptual clusters: {len(all_clusters['perceptual'][0])} assignments")
        print(f"  ✓ Coefficient distributions computed")
        print(f"  ✓ Basis distributions computed")
        
        # Test top features extraction
        top_features_dict = {}
        for data_type in ["acoustic", "perceptual"]:
            if data_type == "acoustic":
                columns = all_data.columns[first_acoustic:]
            else:
                columns = all_data.columns[:first_acoustic]
            
            top_features_dict[data_type] = get_top_features(
                basis_dists[data_type], columns, data_type
            )
        
        print(f"  ✓ Top features extracted for both data types")
        
        print("✅ NMF clustering test passed!")
        return True, (all_clusters, all_feature_maps, coef_dists, raw_coef_dists, basis_dists, top_features_dict)
        
    except Exception as e:
        print(f"  ❌ NMF clustering failed: {str(e)}")
        traceback.print_exc()
        return False, None


def test_visualization(test_data, clustering_data):
    """Test visualization functionality."""
    print("\n🔍 Testing Visualization...")
    
    try:
        from src.visualization import setup_output_directories, plot_pca_clusters
        from sklearn.preprocessing import MinMaxScaler
        
        all_data, clinical_summaries, demographics, first_acoustic, demographic_stats = test_data
        all_clusters, all_feature_maps, coef_dists, basis_dists, top_features_dict = clustering_data
        
        print("  Setting up output directories...")
        setup_output_directories()
        
        print("  Testing PCA cluster visualization...")
        # Test one PCA plot for acoustic data
        V = all_data.iloc[:, first_acoustic:].values
        mms = MinMaxScaler()
        V_scaled = mms.fit_transform(V)
        
        # Create a test plot (but don't save to avoid cluttering)
        plot_pca_clusters(V_scaled, all_clusters["acoustic"], "acoustic", save_path=None, original_data=V_scaled, zero_participants=None)
        
        print("  ✓ PCA visualization test completed")
        print("✅ Visualization test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Visualization failed: {str(e)}")
        traceback.print_exc()
        return False


def test_cross_modal_analysis(test_data, clustering_data):
    """Test cross-modal analysis functionality."""
    print("\n🔍 Testing Cross-Modal Analysis...")
    
    try:
        from src.cross_modal_analysis import run_cross_modal_analysis
        
        all_data, clinical_summaries, demographics, first_acoustic, demographic_stats = test_data
        all_clusters, all_feature_maps, coef_dists, raw_coef_dists, basis_dists, top_features_dict = clustering_data
        
        print("  Running cross-modal analysis...")
        analysis_results, similarity_metrics = run_cross_modal_analysis(
            all_clusters, all_feature_maps, coef_dists, clinical_summaries, raw_coef_dists
        )
        
        print(f"  ✓ Cluster mappings computed")
        print(f"  ✓ Severity statistics calculated")
        print(f"  ✓ Similarity metrics computed")
        print(f"  ✓ JS divergences: {len(similarity_metrics['js_divergences'])} values")
        print(f"  ✓ Cosine similarities: {len(similarity_metrics['cosine_similarities'])} values")
        
        print("✅ Cross-modal analysis test passed!")
        return True, (analysis_results, similarity_metrics)
        
    except Exception as e:
        print(f"  ❌ Cross-modal analysis failed: {str(e)}")
        traceback.print_exc()
        return False, None


def test_results_export(test_data, clustering_data, cross_modal_data):
    """Test results export functionality."""
    print("\n🔍 Testing Results Export...")
    
    try:
        from src.results_export import export_all_results
        
        all_data, clinical_summaries, demographics, first_acoustic, demographic_stats = test_data
        all_clusters, all_feature_maps, coef_dists, basis_dists, top_features_dict = clustering_data
        analysis_results, similarity_metrics = cross_modal_data
        
        print("  Exporting results...")
        export_all_results(
            all_clusters, all_data, first_acoustic, analysis_results,
            similarity_metrics, demographic_stats, top_features_dict
        )
        
        print("  ✓ Results exported successfully")
        print("✅ Results export test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Results export failed: {str(e)}")
        traceback.print_exc()
        return False


def run_quick_test():
    """Run a quick test of critical components."""
    print("🚀 Running Quick Test...")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Data Availability", test_data_availability)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        if not test_func():
            all_passed = False
            break
    
    if all_passed:
        print("\n✅ Quick test PASSED! Pipeline is ready to run.")
        return True
    else:
        print("\n❌ Quick test FAILED! Fix the issues above before running the full pipeline.")
        return False


def run_full_test():
    """Run comprehensive test of entire pipeline."""
    print("🚀 Running Full Pipeline Test...")
    start_time = time.time()
    
    # Test 1: Dependencies and imports
    if not test_dependencies() or not test_imports():
        return False
    
    # Test 2: Data availability and loading
    if not test_data_availability():
        return False
    
    data_success, test_data = test_data_loading()
    if not data_success:
        return False
    
    # Test 3: NMF clustering
    clustering_success, clustering_data = test_nmf_clustering(test_data)
    if not clustering_success:
        return False
    
    # Test 4: Visualization
    if not test_visualization(test_data, clustering_data):
        return False
    
    # Test 5: Cross-modal analysis
    cross_modal_success, cross_modal_data = test_cross_modal_analysis(test_data, clustering_data)
    if not cross_modal_success:
        return False
    
    # Test 6: Results export
    if not test_results_export(test_data, clustering_data, cross_modal_data):
        return False
    
    elapsed_time = time.time() - start_time
    print(f"\n🎉 FULL TEST PASSED! ({elapsed_time:.2f} seconds)")
    print("✅ Your pipeline is working correctly and ready for production use!")
    return True


def main():
    """Main testing function."""
    print("=" * 80)
    print("ALS CLUSTERING ANALYSIS - PIPELINE TESTER")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_test()
    else:
        print("Running full test (use --quick for quick test only)")
        success = run_full_test()
    
    if success:
        print("\n🎯 Next Steps:")
        print("  1. Run the full pipeline: python run_analysis.py")
        print("  2. Check outputs in the 'outputs/' directory")
        print("  3. Review the summary report for key findings")
    else:
        print("\n🔧 Fix the issues above before running the full pipeline.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
