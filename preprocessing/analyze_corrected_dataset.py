#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Corrected Dataset Quality

This script analyzes the corrected dataset to identify potential improvements
and ensures high-quality emotion labels.
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter

class DatasetQualityAnalyzer:
    """Analyzes dataset quality and suggests improvements"""
    
    def __init__(self):
        # Define emotion patterns for quality check
        self.strong_negative_patterns = [
            r'दर्द.*[^खुश]', r'गम.*[^खुश]', r'रो.*[^खुश]', r'आंसू', r'ज़ख्म', r'टूट.*दिल',
            r'बिछड़.*[^खुश]', r'जुदाई', r'मौत', r'मर.*जा', r'बर्बाद', r'तबाह', r'निराश'
        ]
        
        self.strong_positive_patterns = [
            r'खुश.*[^दर्द]', r'आनंद', r'मुस्कान', r'हंस.*[^दर्द]', r'जीत', r'सफल', 
            r'उम्मीद.*[^टूट]', r'प्रेम.*[^दर्द]', r'आशा.*[^टूट]'
        ]
        
        self.ambiguous_patterns = [
            r'प्यार.*दर्द', r'दर्द.*प्यार', r'इश्क़.*गम', r'मोहब्बत.*तकलीफ'
        ]

    def analyze_emotion_consistency(self, df):
        """Analyze emotion label consistency"""
        print("🔍 ANALYZING EMOTION LABEL CONSISTENCY")
        print("=" * 60)
        
        inconsistencies = []
        
        for i, row in df.iterrows():
            text = str(row['text']).lower()
            emotion = row['emotion']
            
            # Check for strong patterns that contradict labels
            strong_neg_found = any(re.search(pattern, text) for pattern in self.strong_negative_patterns)
            strong_pos_found = any(re.search(pattern, text) for pattern in self.strong_positive_patterns)
            ambiguous_found = any(re.search(pattern, text) for pattern in self.ambiguous_patterns)
            
            issue = None
            
            # Flag potential inconsistencies
            if strong_neg_found and emotion == 'positive':
                issue = "Strong negative words but labeled positive"
            elif strong_pos_found and emotion == 'negative':
                issue = "Strong positive words but labeled negative"
            elif ambiguous_found and emotion == 'positive':
                issue = "Pain+love pattern should likely be negative"
                
            if issue:
                inconsistencies.append({
                    'index': i,
                    'text': row['text'][:100],
                    'emotion': emotion,
                    'issue': issue,
                    'strong_neg': strong_neg_found,
                    'strong_pos': strong_pos_found,
                    'ambiguous': ambiguous_found
                })
        
        print(f"📊 CONSISTENCY ANALYSIS:")
        print(f"  Total samples: {len(df)}")
        print(f"  Potential inconsistencies: {len(inconsistencies)}")
        print(f"  Consistency rate: {(len(df) - len(inconsistencies))/len(df)*100:.1f}%")
        
        if inconsistencies:
            print(f"\n⚠️ POTENTIAL INCONSISTENCIES (Top 10):")
            for inc in inconsistencies[:10]:
                print(f"  📄 {inc['text']}...")
                print(f"      Labeled: {inc['emotion']}, Issue: {inc['issue']}")
                print()
        
        return inconsistencies

    def analyze_class_balance_quality(self, df):
        """Analyze quality of class balance"""
        print(f"\n📊 CLASS BALANCE QUALITY ANALYSIS")
        print("=" * 60)
        
        emotion_counts = df['emotion'].value_counts()
        total = len(df)
        
        print(f"Current distribution:")
        for emotion, count in emotion_counts.items():
            percentage = (count / total) * 100
            print(f"  {emotion}: {count} samples ({percentage:.1f}%)")
        
        # Calculate balance score
        expected_per_class = total / len(emotion_counts)
        balance_score = 1 - np.std([count for count in emotion_counts.values()]) / expected_per_class
        
        print(f"\n⚖️ BALANCE METRICS:")
        print(f"  Expected per class: {expected_per_class:.1f}")
        print(f"  Balance score: {balance_score:.3f} (1.0 = perfect balance)")
        
        return balance_score

    def analyze_text_quality(self, df):
        """Analyze text quality and characteristics"""
        print(f"\n📝 TEXT QUALITY ANALYSIS")
        print("=" * 60)
        
        # Text length analysis
        df['text_length'] = df['text'].str.len()
        
        print(f"Text length statistics:")
        print(f"  Average: {df['text_length'].mean():.1f} characters")
        print(f"  Median: {df['text_length'].median():.1f} characters")
        print(f"  Min: {df['text_length'].min()} characters")
        print(f"  Max: {df['text_length'].max()} characters")
        print(f"  Std: {df['text_length'].std():.1f} characters")
        
        # Check for very short or long texts
        short_texts = df[df['text_length'] < 30]
        long_texts = df[df['text_length'] > 200]
        
        print(f"\n📏 TEXT LENGTH DISTRIBUTION:")
        print(f"  Very short (<30 chars): {len(short_texts)} texts")
        print(f"  Normal (30-200 chars): {len(df) - len(short_texts) - len(long_texts)} texts")
        print(f"  Very long (>200 chars): {len(long_texts)} texts")
        
        if len(short_texts) > 0:
            print(f"\n⚠️ VERY SHORT TEXTS:")
            for i, row in short_texts.head(5).iterrows():
                print(f"  '{row['text']}' -> {row['emotion']}")
        
        # Analyze vocabulary diversity
        all_words = ' '.join(df['text']).lower().split()
        unique_words = set(all_words)
        
        print(f"\n📚 VOCABULARY ANALYSIS:")
        print(f"  Total words: {len(all_words)}")
        print(f"  Unique words: {len(unique_words)}")
        print(f"  Vocabulary diversity: {len(unique_words)/len(all_words):.3f}")
        
        return {
            'avg_length': df['text_length'].mean(),
            'vocab_diversity': len(unique_words)/len(all_words),
            'short_texts': len(short_texts),
            'long_texts': len(long_texts)
        }

    def suggest_improvements(self, df, inconsistencies, quality_metrics):
        """Suggest dataset improvements"""
        print(f"\n💡 DATASET IMPROVEMENT SUGGESTIONS")
        print("=" * 60)
        
        suggestions = []
        
        # Check inconsistencies
        if len(inconsistencies) > len(df) * 0.05:  # More than 5% inconsistent
            suggestions.append(f"🔧 Fix {len(inconsistencies)} potential label inconsistencies")
        
        # Check text length issues
        if quality_metrics['short_texts'] > 5:
            suggestions.append(f"📏 Review {quality_metrics['short_texts']} very short texts")
        
        if quality_metrics['long_texts'] > 10:
            suggestions.append(f"📏 Consider truncating {quality_metrics['long_texts']} very long texts")
        
        # Check vocabulary diversity
        if quality_metrics['vocab_diversity'] < 0.1:
            suggestions.append("📚 Low vocabulary diversity - consider more diverse samples")
        
        # Check balance
        emotion_counts = df['emotion'].value_counts()
        min_count = emotion_counts.min()
        max_count = emotion_counts.max()
        
        if max_count / min_count > 1.2:  # More than 20% imbalance
            suggestions.append("⚖️ Minor class imbalance detected")
        
        if suggestions:
            print("Recommended improvements:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        else:
            print("✅ Dataset quality is good! No major improvements needed.")
        
        return suggestions

def analyze_model_performance_vs_labels(df):
    """Analyze how well model performance aligns with label quality"""
    print(f"\n🎯 MODEL PERFORMANCE vs LABEL QUALITY")
    print("=" * 60)
    
    # Sample analysis for each emotion
    emotions = df['emotion'].unique()
    
    for emotion in emotions:
        emotion_samples = df[df['emotion'] == emotion]
        print(f"\n{emotion.upper()} SAMPLES ({len(emotion_samples)} total):")
        
        # Show representative samples
        sample_texts = emotion_samples['text'].head(3)
        for i, text in enumerate(sample_texts, 1):
            print(f"  {i}. {text[:120]}...")
        
        # Basic keyword analysis
        all_text = ' '.join(emotion_samples['text']).lower()
        
        # Count emotional keywords
        negative_words = ['दर्द', 'गम', 'आंसू', 'ज़ख्म', 'टूट', 'बिछड़', 'जुदाई']
        positive_words = ['खुश', 'आनंद', 'मुस्कान', 'हंस', 'प्रेम', 'उम्मीद', 'सफल']
        
        neg_count = sum(all_text.count(word) for word in negative_words)
        pos_count = sum(all_text.count(word) for word in positive_words)
        
        print(f"     Negative keywords: {neg_count}, Positive keywords: {pos_count}")

def main():
    """Main analysis function"""
    print("🔬 CORRECTED DATASET QUALITY ANALYSIS")
    print("=" * 70)
    
    # Load corrected dataset
    dataset_file = "datasets/output_with_emotions_corrected_balanced_20250609_140452.xlsx"
    df = pd.read_excel(dataset_file)
    
    print(f"📂 Dataset: {dataset_file}")
    print(f"📊 Size: {df.shape}")
    print(f"📝 Columns: {df.columns.tolist()}")
    
    # Initialize analyzer
    analyzer = DatasetQualityAnalyzer()
    
    # Run analyses
    inconsistencies = analyzer.analyze_emotion_consistency(df)
    balance_score = analyzer.analyze_class_balance_quality(df)
    quality_metrics = analyzer.analyze_text_quality(df)
    suggestions = analyzer.suggest_improvements(df, inconsistencies, quality_metrics)
    
    # Analyze model performance context
    analyze_model_performance_vs_labels(df)
    
    # Overall quality score
    print(f"\n📈 OVERALL DATASET QUALITY SCORE")
    print("=" * 60)
    
    consistency_score = (len(df) - len(inconsistencies)) / len(df)
    length_score = 1 - (quality_metrics['short_texts'] + quality_metrics['long_texts']) / len(df)
    vocab_score = min(quality_metrics['vocab_diversity'] * 10, 1.0)  # Cap at 1.0
    
    overall_score = (consistency_score + balance_score + length_score + vocab_score) / 4
    
    print(f"  Consistency Score: {consistency_score:.3f}")
    print(f"  Balance Score: {balance_score:.3f}")
    print(f"  Length Score: {length_score:.3f}")
    print(f"  Vocabulary Score: {vocab_score:.3f}")
    print(f"  OVERALL SCORE: {overall_score:.3f} / 1.000")
    
    if overall_score >= 0.8:
        quality_rating = "EXCELLENT 🌟"
    elif overall_score >= 0.7:
        quality_rating = "GOOD ✅"
    elif overall_score >= 0.6:
        quality_rating = "ACCEPTABLE ⚠️"
    else:
        quality_rating = "NEEDS IMPROVEMENT ❌"
    
    print(f"  Quality Rating: {quality_rating}")
    
    # Final recommendation
    print(f"\n🎯 FINAL RECOMMENDATION:")
    if overall_score >= 0.75:
        print("✅ Dataset quality is good enough for model training!")
        print("🚀 Proceed with training all models on this corrected dataset.")
    else:
        print("⚠️ Consider addressing the suggested improvements before final training.")
    
    return overall_score, suggestions

if __name__ == "__main__":
    score, improvements = main() 