# Combined Model Results Analysis

This folder contains comprehensive analysis and comparison of all emotion classification models.

## Generated Files

### 📊 Visualizations

- **`comprehensive_model_comparison.png`** - 2x2 subplot showing overall metrics, precision, recall, and F1-scores by emotion
- **`overall_performance_comparison.png`** - Multi-line graph with models as lines across key performance metrics
- **`emotion_wise_performance_comparison.png`** - 2x2 subplot showing emotion-specific performance comparisons
- **`model_performance_radar.png`** - Radar chart comparing models across multiple metrics

### 📄 Data Files

- **`detailed_model_comparison.csv`** - Complete numerical data for all metrics across all models

### 🔧 Scripts

- **`results_comparison_analysis.py`** - Comprehensive analysis script with multiple visualization types
- **`comprehensive_model_comparison.py`** - Focused script for the main comparison visualization

## Model Performance Summary

### Overall Ranking (by Accuracy)

1. **MultiBERT** - 65.3% accuracy, 65.5% F1, 83.5% AUC
2. **BlueBERT_BIO** - 52.1% accuracy, 52.3% F1, 70.0% AUC
3. **BioBERT_BIO** - 50.0% accuracy, 50.8% F1, 71.2% AUC
4. **BlueBERT** - 35.4% accuracy, 26.2% F1, 61.0% AUC
5. **BioBERT** - 33.3% accuracy, 31.0% F1, 49.1% AUC

### Key Insights

- MultiBERT significantly outperforms all other models
- BIO-enhanced models perform better than their base versions
- BlueBERT_BIO shows best negative emotion detection
- Significant variation in emotion-specific performance across models

## Usage

To regenerate the analysis:

```bash
# Run comprehensive analysis
python results_comparison_analysis.py

# Or run focused comparison
python comprehensive_model_comparison.py
```

Both scripts will automatically generate outputs in the current directory.

## Requirements

- pandas>=1.5.0
- matplotlib>=3.6.0
- seaborn>=0.11.0
- numpy>=1.21.0
