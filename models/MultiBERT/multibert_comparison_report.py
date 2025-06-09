#!/usr/bin/env python3
"""
Comprehensive Comparison Report: MultiBERT vs BioBERT/BlueBERT
Analyzing performance across all emotion classification models
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_model_results():
    """Load results from all trained models"""
    results = {}
    
    # MultiBERT results
    multibert_path = "results/multibert_metrics.json"
    multibert_hindi_path = "results/multibert_hindi_metrics.json"
    
    # BioBERT/BlueBERT results
    biobert_path = "../BioBERT/results/biobert_metrics.json"
    biobert_bio_path = "../BioBERT_BIO/results/biobert_bio_metrics.json"
    bluebert_path = "../BlueBERT/results/bluebert_metrics.json"
    bluebert_bio_path = "../BlueBERT_BIO/results/bluebert_bio_metrics.json"
    
    model_files = {
        "MultiBERT": multibert_path,
        "MultiBERT + Hindi": multibert_hindi_path,
        "BioBERT": biobert_path,
        "BioBERT + Hindi": biobert_bio_path,
        "BlueBERT": bluebert_path,
        "BlueBERT + Hindi": bluebert_bio_path
    }
    
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    results[model_name] = json.load(f)
                print(f"✅ Loaded {model_name} results")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {str(e)}")
        else:
            print(f"⚠️  {model_name} results not found at {file_path}")
    
    return results

def create_comparison_dataframe(results):
    """Create a comprehensive comparison DataFrame"""
    metrics_data = []
    
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0),
            'F1 (Macro)': metrics.get('f1_macro', 0),
            'F1 (Weighted)': metrics.get('f1_weighted', 0),
            'Precision (Macro)': metrics.get('precision_macro', 0),
            'Precision (Weighted)': metrics.get('precision_weighted', 0),
            'Recall (Macro)': metrics.get('recall_macro', 0),
            'Recall (Weighted)': metrics.get('recall_weighted', 0),
            'AUC-ROC': metrics.get('auc_roc', 0),
            'MCC': metrics.get('mcc', 0),
            'RMSE': metrics.get('rmse', 0)
        }
        
        # Add emotion-wise F1 scores
        if 'per_class_metrics' in metrics:
            for emotion in ['negative', 'neutral', 'positive']:
                if emotion in metrics['per_class_metrics']:
                    row[f'{emotion.title()} F1'] = metrics['per_class_metrics'][emotion]['f1-score']
                else:
                    row[f'{emotion.title()} F1'] = 0
        
        metrics_data.append(row)
    
    return pd.DataFrame(metrics_data)

def print_comparison_table(df):
    """Print a formatted comparison table"""
    print("\n" + "="*120)
    print("🔍 COMPREHENSIVE MODEL COMPARISON")
    print("="*120)
    
    # Sort by accuracy descending
    df_sorted = df.sort_values('Accuracy', ascending=False)
    
    print(f"\n📊 OVERALL PERFORMANCE RANKING:")
    print("-" * 120)
    print(f"{'Rank':<4} {'Model':<20} {'Accuracy':<10} {'F1 Macro':<10} {'F1 Weighted':<12} {'AUC-ROC':<8} {'MCC':<8}")
    print("-" * 120)
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{i:<4} {row['Model']:<20} {row['Accuracy']:<10.4f} {row['F1 (Macro)']:<10.4f} {row['F1 (Weighted)']:<12.4f} {row['AUC-ROC']:<8.4f} {row['MCC']:<8.4f}")
    
    print("\n📈 EMOTION-WISE PERFORMANCE:")
    print("-" * 80)
    print(f"{'Model':<20} {'Negative F1':<12} {'Neutral F1':<12} {'Positive F1':<12}")
    print("-" * 80)
    
    for _, row in df_sorted.iterrows():
        neg_f1 = row.get('Negative F1', 0)
        neu_f1 = row.get('Neutral F1', 0)
        pos_f1 = row.get('Positive F1', 0)
        print(f"{row['Model']:<20} {neg_f1:<12.4f} {neu_f1:<12.4f} {pos_f1:<12.4f}")

def create_performance_plots(df):
    """Create comprehensive performance visualization plots"""
    
    # Set style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Overall Metrics Comparison
    ax1 = plt.subplot(2, 3, 1)
    metrics_cols = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'AUC-ROC']
    x_pos = range(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics_cols):
        plt.bar([x + i*width for x in x_pos], df[metric], width, 
                label=metric, alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Overall Performance Comparison')
    plt.xticks([x + width*1.5 for x in x_pos], df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy Bar Chart
    ax2 = plt.subplot(2, 3, 2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = plt.bar(df['Model'], df['Accuracy'], color=colors[:len(df)], alpha=0.8)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 3. F1 Scores Comparison
    ax3 = plt.subplot(2, 3, 3)
    f1_cols = ['F1 (Macro)', 'F1 (Weighted)']
    x_pos = range(len(df))
    width = 0.35
    
    for i, f1_metric in enumerate(f1_cols):
        plt.bar([x + i*width for x in x_pos], df[f1_metric], width, 
                label=f1_metric, alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison')
    plt.xticks([x + width/2 for x in x_pos], df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Emotion-wise F1 Scores
    ax4 = plt.subplot(2, 3, 4)
    emotion_cols = ['Negative F1', 'Neutral F1', 'Positive F1']
    x_pos = range(len(df))
    width = 0.25
    
    for i, emotion in enumerate(emotion_cols):
        if emotion in df.columns:
            plt.bar([x + i*width for x in x_pos], df[emotion], width, 
                    label=emotion.replace(' F1', ''), alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('F1 Score')
    plt.title('Emotion-wise F1 Score Comparison')
    plt.xticks([x + width for x in x_pos], df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. AUC-ROC vs MCC
    ax5 = plt.subplot(2, 3, 5)
    plt.scatter(df['AUC-ROC'], df['MCC'], s=100, alpha=0.7, c=colors[:len(df)])
    
    for i, model in enumerate(df['Model']):
        plt.annotate(model, (df.iloc[i]['AUC-ROC'], df.iloc[i]['MCC']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('AUC-ROC')
    plt.ylabel('Matthews Correlation Coefficient')
    plt.title('AUC-ROC vs MCC')
    plt.grid(True, alpha=0.3)
    
    # 6. Radar Chart for Top Models
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    
    # Select top 3 models by accuracy
    top_models = df.nlargest(3, 'Accuracy')
    
    metrics_radar = ['Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)', 'AUC-ROC']
    angles = [n / float(len(metrics_radar)) * 2 * 3.14159 for n in range(len(metrics_radar))]
    angles += angles[:1]  # Complete the circle
    
    for i, (_, row) in enumerate(top_models.iterrows()):
        values = [row[metric] for metric in metrics_radar]
        values += values[:1]  # Complete the circle
        
        ax6.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
        ax6.fill(angles, values, alpha=0.25)
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics_radar)
    ax6.set_ylim(0, 1)
    ax6.set_title('Top 3 Models - Radar Chart')
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('multibert_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Comprehensive comparison plot saved to: multibert_comprehensive_comparison.png")

def analyze_improvements(df):
    """Analyze improvements between base models and enhanced versions"""
    print("\n📈 FEATURE ENHANCEMENT ANALYSIS:")
    print("="*80)
    
    # Compare base vs enhanced models
    comparisons = [
        ("MultiBERT", "MultiBERT + Hindi"),
        ("BioBERT", "BioBERT + Hindi"),
        ("BlueBERT", "BlueBERT + Hindi")
    ]
    
    for base_model, enhanced_model in comparisons:
        if base_model in df['Model'].values and enhanced_model in df['Model'].values:
            base_acc = df[df['Model'] == base_model]['Accuracy'].iloc[0]
            enhanced_acc = df[df['Model'] == enhanced_model]['Accuracy'].iloc[0]
            
            base_f1 = df[df['Model'] == base_model]['F1 (Macro)'].iloc[0]
            enhanced_f1 = df[df['Model'] == enhanced_model]['F1 (Macro)'].iloc[0]
            
            acc_improvement = (enhanced_acc - base_acc) * 100
            f1_improvement = (enhanced_f1 - base_f1) * 100
            
            print(f"\n{base_model} → {enhanced_model}:")
            print(f"  Accuracy: {base_acc:.4f} → {enhanced_acc:.4f} ({acc_improvement:+.2f}%)")
            print(f"  F1 Macro: {base_f1:.4f} → {enhanced_f1:.4f} ({f1_improvement:+.2f}%)")
            
            if acc_improvement > 0:
                print(f"  ✅ Improvement with enhanced features")
            else:
                print(f"  ❌ Degradation with enhanced features")

def generate_insights(df):
    """Generate key insights from the comparison"""
    print("\n🔍 KEY INSIGHTS:")
    print("="*80)
    
    # Best overall model
    best_model = df.loc[df['Accuracy'].idxmax()]
    print(f"🏆 Best Overall Model: {best_model['Model']}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    print(f"   F1 Macro: {best_model['F1 (Macro)']:.4f}")
    print(f"   AUC-ROC: {best_model['AUC-ROC']:.4f}")
    
    # Compare MultiBERT vs BioBERT/BlueBERT
    multibert_models = df[df['Model'].str.contains('MultiBERT')]
    biobert_models = df[df['Model'].str.contains('BioBERT')]
    bluebert_models = df[df['Model'].str.contains('BlueBERT')]
    
    if not multibert_models.empty:
        avg_multibert_acc = multibert_models['Accuracy'].mean()
        print(f"\n🌍 MultiBERT Average Accuracy: {avg_multibert_acc:.4f}")
        
        if not biobert_models.empty:
            avg_biobert_acc = biobert_models['Accuracy'].mean()
            print(f"🧬 BioBERT Average Accuracy: {avg_biobert_acc:.4f}")
            
            if avg_multibert_acc > avg_biobert_acc:
                print("   ✅ MultiBERT outperforms BioBERT on average")
            else:
                print("   ❌ BioBERT outperforms MultiBERT on average")
        
        if not bluebert_models.empty:
            avg_bluebert_acc = bluebert_models['Accuracy'].mean()
            print(f"🔵 BlueBERT Average Accuracy: {avg_bluebert_acc:.4f}")
            
            if avg_multibert_acc > avg_bluebert_acc:
                print("   ✅ MultiBERT outperforms BlueBERT on average")
            else:
                print("   ❌ BlueBERT outperforms MultiBERT on average")
    
    # Language appropriateness insight
    print(f"\n🇮🇳 LANGUAGE APPROPRIATENESS INSIGHT:")
    print("MultiBERT (bert-base-multilingual-cased) is specifically designed for")
    print("multilingual tasks including Hindi, making it more appropriate for")
    print("Hindi emotion classification than English-only BioBERT/BlueBERT models.")
    
    # Best emotion classification
    emotion_cols = ['Negative F1', 'Neutral F1', 'Positive F1']
    for emotion_col in emotion_cols:
        if emotion_col in df.columns:
            best_emotion_model = df.loc[df[emotion_col].idxmax()]
            emotion_name = emotion_col.replace(' F1', '').lower()
            print(f"\n🎯 Best {emotion_name.title()} Classification: {best_emotion_model['Model']}")
            print(f"   {emotion_col}: {best_emotion_model[emotion_col]:.4f}")

def save_comparison_report(df, results):
    """Save comprehensive comparison report"""
    report_path = "multibert_comparison_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
        f.write("MultiBERT vs BioBERT/BlueBERT for Hindi Emotion Classification\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODELS EVALUATED:\n")
        f.write("-" * 50 + "\n")
        for model in df['Model']:
            f.write(f"• {model}\n")
        
        f.write(f"\nOVERALL PERFORMANCE RANKING:\n")
        f.write("-" * 50 + "\n")
        df_sorted = df.sort_values('Accuracy', ascending=False)
        
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            f.write(f"{i}. {row['Model']}\n")
            f.write(f"   Accuracy: {row['Accuracy']:.4f}\n")
            f.write(f"   F1 Macro: {row['F1 (Macro)']:.4f}\n")
            f.write(f"   AUC-ROC: {row['AUC-ROC']:.4f}\n")
            f.write(f"   MCC: {row['MCC']:.4f}\n\n")
        
        f.write("DETAILED METRICS TABLE:\n")
        f.write("-" * 50 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 50 + "\n")
        best_model = df.loc[df['Accuracy'].idxmax()]
        f.write(f"• Best performing model: {best_model['Model']}\n")
        f.write(f"• Highest accuracy: {best_model['Accuracy']:.4f}\n")
        f.write(f"• MultiBERT is more linguistically appropriate for Hindi text\n")
        f.write(f"• Hindi emotional features provide mixed improvements\n")
        
    print(f"✅ Comprehensive report saved to: {report_path}")

def main():
    print("🔍 MULTIBERT COMPREHENSIVE COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Load all model results
    results = load_model_results()
    
    if not results:
        print("❌ No model results found. Please train models first.")
        return
    
    # Create comparison DataFrame
    df = create_comparison_dataframe(results)
    
    # Print comparison table
    print_comparison_table(df)
    
    # Analyze improvements
    analyze_improvements(df)
    
    # Generate insights
    generate_insights(df)
    
    # Create performance plots
    create_performance_plots(df)
    
    # Save comprehensive report
    save_comparison_report(df, results)
    
    print(f"\n🎉 COMPREHENSIVE COMPARISON ANALYSIS COMPLETE!")
    print(f"📊 Results: {len(results)} models compared")
    print(f"📈 Visualizations: multibert_comprehensive_comparison.png")
    print(f"📄 Report: multibert_comparison_report.txt")

if __name__ == "__main__":
    main() 