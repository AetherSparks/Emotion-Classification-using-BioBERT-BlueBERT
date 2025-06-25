import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
from pathlib import Path

class ModelResultsAnalyzer:
    def __init__(self):
        self.results_data = {}
        self.models = ['BioBERT', 'BlueBERT', 'BioBERT_BIO', 'BlueBERT_BIO', 'MultiBERT', 'MultiBERT_Hindi']
        self.results_files = {
            'BioBERT': '../models/BioBERT/results/biobert_report.txt',
            'BlueBERT': '../models/BlueBERT/results/bluebert_report.txt',
            'BioBERT_BIO': '../models/BioBERT_BIO/results/biobert_bio_report.txt',
            'BlueBERT_BIO': '../models/BlueBERT_BIO/results/bluebert_bio_report.txt',
            'MultiBERT': '../models/MultiBERT/results/multibert_report.txt',
            'MultiBERT_Hindi': '../models/MultiBERT/results/multibert_hindi_report.txt'
        }
        
        # Create output directory
        self.output_dir = Path('.')  # Current directory since we're already in combinedresults
        self.output_dir.mkdir(exist_ok=True)
        
    def parse_results_file(self, filepath):
        """Parse a results file and extract metrics"""
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found")
            return None
            
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract overall metrics
        metrics = {}
        
        # Overall metrics patterns
        patterns = {
            'Accuracy': r'Accuracy:\s+([\d.]+)',
            'F1_Macro': r'F1 Score \(Macro\):\s+([\d.]+)',
            'F1_Weighted': r'F1 Score \(Weighted\):\s+([\d.]+)',
            'Precision_Macro': r'Precision \(Macro\):\s+([\d.]+)',
            'Precision_Weighted': r'Precision \(Weighted\):\s+([\d.]+)',
            'Recall_Macro': r'Recall \(Macro\):\s+([\d.]+)',
            'Recall_Weighted': r'Recall \(Weighted\):\s+([\d.]+)',
            'RMSE': r'RMSE:\s+([\d.]+)',
            'AUC_ROC': r'AUC-ROC:\s+([\d.]+)',
            'MCC': r'MCC:\s+([\d.]+)'
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[metric] = float(match.group(1))
            else:
                metrics[metric] = 0.0
        
        # Extract emotion-wise results
        emotions = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        emotion_metrics = {}
        
        for emotion in emotions:
            emotion_section = re.search(rf'{emotion}:\s*\n\s*Precision:\s*([\d.]+)\s*\n\s*Recall:\s*([\d.]+)\s*\n\s*F1-Score:\s*([\d.]+)\s*\n\s*Support:\s*([\d.]+)', content)
            if emotion_section:
                emotion_metrics[f'{emotion}_Precision'] = float(emotion_section.group(1))
                emotion_metrics[f'{emotion}_Recall'] = float(emotion_section.group(2))
                emotion_metrics[f'{emotion}_F1'] = float(emotion_section.group(3))
                emotion_metrics[f'{emotion}_Support'] = float(emotion_section.group(4))
            else:
                emotion_metrics[f'{emotion}_Precision'] = 0.0
                emotion_metrics[f'{emotion}_Recall'] = 0.0
                emotion_metrics[f'{emotion}_F1'] = 0.0
                emotion_metrics[f'{emotion}_Support'] = 0.0
        
        # Combine all metrics
        all_metrics = {**metrics, **emotion_metrics}
        return all_metrics
    
    def load_all_results(self):
        """Load results from all model files"""
        for model, filepath in self.results_files.items():
            results = self.parse_results_file(filepath)
            if results:
                self.results_data[model] = results
                print(f"✓ Loaded results for {model}")
            else:
                print(f"✗ Failed to load results for {model}")
    
    def create_comparison_dataframe(self):
        """Create a DataFrame for easy comparison"""
        if not self.results_data:
            self.load_all_results()
        
        df = pd.DataFrame(self.results_data).T
        return df
    
    def plot_overall_performance_metrics(self):
        """Create multi-line graph with models as lines and metrics on x-axis"""
        df = self.create_comparison_dataframe()
        
        # Define metrics to plot
        overall_metrics = ['Accuracy', 'F1_Macro', 'Precision_Macro', 'Recall_Macro', 'AUC_ROC', 'MCC']
        
        plt.figure(figsize=(15, 8))
        
        # Create x positions for metrics
        x_positions = np.arange(len(overall_metrics))
        
        # Colors for each model
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Plot each model as a line
        for i, model in enumerate(self.models):
            if model in df.index:
                values = [df.loc[model, metric] if metric in df.columns else 0 for metric in overall_metrics]
                plt.plot(x_positions, values, marker='o', linewidth=3, markersize=10, 
                        label=model, color=colors[i % len(colors)])
                
                # Add value labels on points
                for j, value in enumerate(values):
                    plt.annotate(f'{value:.3f}', (j, value), textcoords="offset points", 
                               xytext=(0,15), ha='center', fontsize=9, fontweight='bold',
                               color=colors[i % len(colors)])
        
        plt.title('Model Performance Comparison Across Key Metrics', fontsize=16, fontweight='bold')
        plt.xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.xticks(x_positions, overall_metrics, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add horizontal reference lines
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Performance')
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% Performance')
        
        plt.tight_layout()
        output_path = self.output_dir / 'overall_performance_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_emotion_wise_performance(self):
        """Create separate plots for each emotion's performance across models"""
        df = self.create_comparison_dataframe()
        emotions = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison by Emotion', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Colors for each model
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Plot 1: Precision for all emotions
        ax1 = axes_flat[0]
        x_positions = np.arange(len(emotions))
        for i, model in enumerate(self.models):
            if model in df.index:
                values = [df.loc[model, f'{emotion}_Precision'] if f'{emotion}_Precision' in df.columns else 0 for emotion in emotions]
                ax1.plot(x_positions, values, marker='o', linewidth=2.5, markersize=8, 
                        label=model, color=colors[i % len(colors)])
        
        ax1.set_title('Precision by Emotion', fontweight='bold')
        ax1.set_xlabel('Emotions')
        ax1.set_ylabel('Precision Score')
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(emotions)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Recall for all emotions
        ax2 = axes_flat[1]
        for i, model in enumerate(self.models):
            if model in df.index:
                values = [df.loc[model, f'{emotion}_Recall'] if f'{emotion}_Recall' in df.columns else 0 for emotion in emotions]
                ax2.plot(x_positions, values, marker='o', linewidth=2.5, markersize=8, 
                        label=model, color=colors[i % len(colors)])
        
        ax2.set_title('Recall by Emotion', fontweight='bold')
        ax2.set_xlabel('Emotions')
        ax2.set_ylabel('Recall Score')
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(emotions)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: F1-Score for all emotions
        ax3 = axes_flat[2]
        for i, model in enumerate(self.models):
            if model in df.index:
                values = [df.loc[model, f'{emotion}_F1'] if f'{emotion}_F1' in df.columns else 0 for emotion in emotions]
                ax3.plot(x_positions, values, marker='o', linewidth=2.5, markersize=8, 
                        label=model, color=colors[i % len(colors)])
        
        ax3.set_title('F1-Score by Emotion', fontweight='bold')
        ax3.set_xlabel('Emotions')
        ax3.set_ylabel('F1-Score')
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels(emotions)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Average emotion performance per model
        ax4 = axes_flat[3]
        emotion_accuracy = []
        model_names = []
        
        for model in self.models:
            if model in df.index:
                # Calculate average F1 score across emotions
                avg_f1 = np.mean([df.loc[model, f'{emotion}_F1'] for emotion in emotions])
                emotion_accuracy.append(avg_f1)
                model_names.append(model)
        
        bars = ax4.bar(range(len(model_names)), emotion_accuracy, color=colors[:len(model_names)])
        ax4.set_title('Average F1-Score Across All Emotions', fontweight='bold')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Average F1-Score')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, emotion_accuracy):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'emotion_wise_performance_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_metrics_table(self):
        """Create a comprehensive comparison table"""
        df = self.create_comparison_dataframe()
        
        # Select key metrics for the table
        key_metrics = [
            'Accuracy', 'F1_Macro', 'AUC_ROC', 'MCC',
            'NEGATIVE_Precision', 'NEGATIVE_Recall', 'NEGATIVE_F1',
            'NEUTRAL_Precision', 'NEUTRAL_Recall', 'NEUTRAL_F1',
            'POSITIVE_Precision', 'POSITIVE_Recall', 'POSITIVE_F1'
        ]
        
        # Create a formatted table
        table_df = df[key_metrics].round(4)
        
        # Add ranking for each metric (1 = best)
        ranking_df = table_df.rank(method='dense', ascending=False)
        
        print("="*80)
        print("COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print("\nPERFORMANCE METRICS:")
        print(table_df.to_string())
        
        print("\n\nRANKING (1 = Best Performance):")
        print(ranking_df.astype(int).to_string())
        
        # Calculate overall ranking score (lower is better)
        overall_ranking = ranking_df.mean(axis=1).sort_values()
        print("\n\nOVERALL MODEL RANKING (Based on Average Rank):")
        for i, (model, rank) in enumerate(overall_ranking.items(), 1):
            print(f"{i}. {model}: {rank:.2f}")
        
        # Save detailed results to CSV
        detailed_results = df.round(4)
        output_path = self.output_dir / 'detailed_model_comparison.csv'
        detailed_results.to_csv(output_path)
        print(f"\n✓ Detailed results saved to '{output_path}'")
        
        return table_df, ranking_df
    
    def plot_radar_chart(self):
        """Create radar chart for model comparison"""
        df = self.create_comparison_dataframe()
        
        # Select metrics for radar chart
        radar_metrics = ['Accuracy', 'F1_Macro', 'Precision_Macro', 'Recall_Macro', 'AUC_ROC']
        
        # Number of variables
        num_vars = len(radar_metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Colors for each model
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Plot each model
        for i, model in enumerate(self.models):
            if model in df.index:
                values = [df.loc[model, metric] if metric in df.columns else 0 for metric in radar_metrics]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        output_path = self.output_dir / 'model_performance_radar.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_full_report(self):
        """Generate complete analysis report"""
        print("🔍 Loading model results...")
        self.load_all_results()
        
        print("\n📊 Creating performance visualizations...")
        self.plot_overall_performance_metrics()
        self.plot_emotion_wise_performance()
        self.plot_radar_chart()
        
        print("\n📋 Generating comparison table...")
        table_df, ranking_df = self.create_comprehensive_metrics_table()
        
        print("\n✅ Analysis complete! Generated files in 'combinedresults' folder:")
        print("   - overall_performance_comparison.png")
        print("   - emotion_wise_performance_comparison.png") 
        print("   - model_performance_radar.png")
        print("   - detailed_model_comparison.csv")

if __name__ == "__main__":
    analyzer = ModelResultsAnalyzer()
    analyzer.generate_full_report() 