import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from pathlib import Path

def load_model_results():
    """Load results from all model files"""
    results_files = {
        'BioBERT': '../models/BioBERT/results/biobert_report.txt',
        'BlueBERT': '../models/BlueBERT/results/bluebert_report.txt',
        'BioBERT_BIO': '../models/BioBERT_BIO/results/biobert_bio_report.txt',
        'BlueBERT_BIO': '../models/BlueBERT_BIO/results/bluebert_bio_report.txt',
        'MultiBERT': '../models/MultiBERT/results/multibert_report.txt',
        'MultiBERT_Hindi': '../models/MultiBERT/results/multibert_hindi_report.txt'
    }
    
    results_data = {}
    
    for model, filepath in results_files.items():
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found")
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract overall metrics
        metrics = {}
        patterns = {
            'Accuracy': r'Accuracy:\s+([\d.]+)',
            'F1_Macro': r'F1 Score \(Macro\):\s+([\d.]+)',
            'Precision_Macro': r'Precision \(Macro\):\s+([\d.]+)',
            'Recall_Macro': r'Recall \(Macro\):\s+([\d.]+)',
            'AUC_ROC': r'AUC-ROC:\s+([\d.]+)',
            'MCC': r'MCC:\s+([\d.]+)'
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, content)
            metrics[metric] = float(match.group(1)) if match else 0.0
        
        # Extract emotion-wise results
        emotions = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        for emotion in emotions:
            emotion_section = re.search(rf'{emotion}:\s*\n\s*Precision:\s*([\d.]+)\s*\n\s*Recall:\s*([\d.]+)\s*\n\s*F1-Score:\s*([\d.]+)', content)
            if emotion_section:
                metrics[f'{emotion}_Precision'] = float(emotion_section.group(1))
                metrics[f'{emotion}_Recall'] = float(emotion_section.group(2))
                metrics[f'{emotion}_F1'] = float(emotion_section.group(3))
            else:
                metrics[f'{emotion}_Precision'] = 0.0
                metrics[f'{emotion}_Recall'] = 0.0
                metrics[f'{emotion}_F1'] = 0.0
        
        results_data[model] = metrics
        print(f"✓ Loaded results for {model}")
    
    return pd.DataFrame(results_data).T

def create_comprehensive_comparison():
    """Create comprehensive model comparison visualizations"""
    df = load_model_results()
    
    # Create output directory (current directory since we're in combinedresults)
    output_dir = Path('.')
    
    # Define different metric groups
    overall_metrics = ['Accuracy', 'F1_Macro', 'Precision_Macro', 'Recall_Macro', 'AUC_ROC', 'MCC']
    precision_metrics = ['NEGATIVE_Precision', 'NEUTRAL_Precision', 'POSITIVE_Precision']
    recall_metrics = ['NEGATIVE_Recall', 'NEUTRAL_Recall', 'POSITIVE_Recall']
    f1_metrics = ['NEGATIVE_F1', 'NEUTRAL_F1', 'POSITIVE_F1']
    
    # Colors for models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Comprehensive Model Performance Comparison', fontsize=18, fontweight='bold')
    
    # Plot 1: Overall Performance Metrics
    ax1 = axes[0, 0]
    x_pos = np.arange(len(overall_metrics))
    for i, model in enumerate(df.index):
        values = [df.loc[model, metric] for metric in overall_metrics]
        ax1.plot(x_pos, values, marker='o', linewidth=3, markersize=8, 
                label=model, color=colors[i % len(colors)])
    
    ax1.set_title('Overall Performance Metrics', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(overall_metrics, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Precision by Emotion
    ax2 = axes[0, 1]
    emotion_labels = ['Negative', 'Neutral', 'Positive']
    x_pos = np.arange(len(precision_metrics))
    for i, model in enumerate(df.index):
        values = [df.loc[model, metric] for metric in precision_metrics]
        ax2.plot(x_pos, values, marker='s', linewidth=3, markersize=8, 
                label=model, color=colors[i % len(colors)])
    
    ax2.set_title('Precision by Emotion Class', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Emotion Classes')
    ax2.set_ylabel('Precision Score')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(emotion_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Recall by Emotion
    ax3 = axes[1, 0]
    x_pos = np.arange(len(recall_metrics))
    for i, model in enumerate(df.index):
        values = [df.loc[model, metric] for metric in recall_metrics]
        ax3.plot(x_pos, values, marker='^', linewidth=3, markersize=8, 
                label=model, color=colors[i % len(colors)])
    
    ax3.set_title('Recall by Emotion Class', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Emotion Classes')
    ax3.set_ylabel('Recall Score')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(emotion_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: F1-Score by Emotion
    ax4 = axes[1, 1]
    x_pos = np.arange(len(f1_metrics))
    for i, model in enumerate(df.index):
        values = [df.loc[model, metric] for metric in f1_metrics]
        ax4.plot(x_pos, values, marker='D', linewidth=3, markersize=8, 
                label=model, color=colors[i % len(colors)])
    
    ax4.set_title('F1-Score by Emotion Class', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Emotion Classes')
    ax4.set_ylabel('F1-Score')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(emotion_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = output_dir / 'comprehensive_model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary statistics
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    for model in df.index:
        print(f"\n{model}:")
        print(f"  Overall Accuracy: {df.loc[model, 'Accuracy']:.3f}")
        print(f"  Macro F1-Score:   {df.loc[model, 'F1_Macro']:.3f}")
        print(f"  AUC-ROC:          {df.loc[model, 'AUC_ROC']:.3f}")
        print(f"  MCC:              {df.loc[model, 'MCC']:.3f}")
        
        # Best emotion performance
        neg_f1 = df.loc[model, 'NEGATIVE_F1']
        neu_f1 = df.loc[model, 'NEUTRAL_F1']
        pos_f1 = df.loc[model, 'POSITIVE_F1']
        best_emotion = ['Negative', 'Neutral', 'Positive'][np.argmax([neg_f1, neu_f1, pos_f1])]
        print(f"  Best Emotion:     {best_emotion} (F1: {max(neg_f1, neu_f1, pos_f1):.3f})")
    
    # Overall ranking
    print(f"\n{'='*80}")
    print("OVERALL MODEL RANKING (by Accuracy)")
    print("="*80)
    ranking = df.sort_values('Accuracy', ascending=False)
    for i, (model, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {model:<15} - Accuracy: {row['Accuracy']:.3f}, F1: {row['F1_Macro']:.3f}, AUC: {row['AUC_ROC']:.3f}")
    
    print(f"\n✅ Files saved in 'combinedresults' folder:")
    print(f"   - comprehensive_model_comparison.png")

if __name__ == "__main__":
    create_comprehensive_comparison() 