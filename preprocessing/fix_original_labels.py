#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Original Dataset Labels and Re-balance

This script fixes emotion labels on the original full dataset (1000 samples)
and then creates a properly balanced dataset with correct labels.
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from datetime import datetime

class HindiUrduEmotionCorrector:
    """Corrects emotion labels for Hindi/Urdu text using linguistic analysis"""
    
    def __init__(self):
        # Define Hindi/Urdu emotion keywords
        self.negative_keywords = {
            # Sadness/Pain
            'दर्द', 'गम', 'सदमा', 'ग़म', 'अफ़सोस', 'आंसू', 'रुलाना', 'रोना', 'तकलीफ',
            'परेशानी', 'डर', 'ख़ौफ़', 'चिंता', 'बेचैनी', 'उदासी', 'अकेलापन',
            # Separation/Loss
            'जुदाई', 'विछोह', 'बिछड़ना', 'खोना', 'मौत', 'मरना', 'मृत्यु', 'अंत',
            'छोड़ना', 'त्यागना', 'भूलना', 'भुलाना', 'दूर', 'अलग',
            # Negative emotions
            'ग़ुस्सा', 'क्रोध', 'नफ़रत', 'घृणा', 'ईर्ष्या', 'जलन', 'रंजिश', 'शिकायत',
            'बर्बाद', 'तबाह', 'नष्ट', 'खत्म', 'समाप्त', 'नाकाम', 'असफल',
            # Pain words
            'ज़ख्म', 'घाव', 'चोट', 'कष्ट', 'पीड़ा', 'व्यथा', 'संताप', 'यातना',
            # Broken heart
            'टूटा', 'बिखरा', 'शिकस्त', 'हार', 'निराशा', 'हताशा', 'बेकार',
            # Additional negative words
            'दुखी', 'उदास', 'परेशान', 'बेचैन', 'चिंतित', 'डरा', 'भयभीत'
        }
        
        self.positive_keywords = {
            # Happiness/Joy
            'खुशी', 'प्रसन्न', 'आनंद', 'हर्ष', 'उल्लास', 'मस्ती', 'मजा', 'धन्य',
            'सुखी', 'संतुष्ट', 'प्रफुल्लित', 'हंसना', 'मुस्कान', 'मुस्कुराना',
            # Love (positive context)
            'खुशियां', 'जश्न', 'मनाना', 'उत्सव', 'त्योहार', 'सफल', 'जीत', 'विजय',
            'कामयाब', 'सफलता', 'उपलब्धि', 'प्राप्ति', 'पूर्ण', 'संपूर्ण',
            # Hope/Optimism  
            'उम्मीद', 'आशा', 'विश्वास', 'भरोसा', 'हौसला', 'साहस', 'दम', 'जोश',
            'उत्साह', 'जुनून', 'शौक', 'चाह', 'इच्छा', 'कामना', 'अरमान',
            # Additional positive words
            'प्रेम', 'स्नेह', 'वात्सल्य', 'करुणा', 'दया', 'कृपा'
        }
        
        self.neutral_keywords = {
            # Contemplation/Philosophy
            'सोच', 'विचार', 'चिंतन', 'मनन', 'समझ', 'बुद्धि', 'ज्ञान', 'समझदारी',
            'याद', 'स्मृति', 'यादें', 'बीता', 'समय', 'वक़्त', 'काल', 'युग',
            # Neutral states
            'शांत', 'स्थिर', 'संयम', 'धैर्य', 'सब्र', 'इंतज़ार', 'प्रतीक्षा',
            'सामान्य', 'आम', 'साधारण', 'नियमित', 'रोज़ाना', 'दैनिक'
        }
        
        # Context patterns that modify sentiment
        self.pain_love_patterns = [
            r'दर्द.*प्यार', r'प्यार.*दर्द', r'इश्क़.*गम', r'मोहब्बत.*तकलीफ',
            r'दिल.*टूट', r'दिल.*दर्द', r'आंसू.*प्यार', r'ज़ख्म.*इश्क़',
            r'गम.*मोहब्बत', r'तकलीफ.*प्रेम'
        ]
        
        self.positive_love_patterns = [
            r'खुश.*प्यार', r'प्यार.*खुशी', r'मुस्कुराना.*प्यार', r'हंसना.*इश्क़',
            r'आनंद.*प्रेम', r'खुशियां.*मोहब्बत'
        ]
        
    def analyze_emotion_content(self, text):
        """Analyze emotion content using keyword matching and context"""
        text = str(text).lower()
        
        # Count emotion keywords
        negative_count = sum(1 for word in self.negative_keywords if word in text)
        positive_count = sum(1 for word in self.positive_keywords if word in text)
        neutral_count = sum(1 for word in self.neutral_keywords if word in text)
        
        # Check for pain+love patterns (typically negative)
        pain_love_found = any(re.search(pattern, text) for pattern in self.pain_love_patterns)
        positive_love_found = any(re.search(pattern, text) for pattern in self.positive_love_patterns)
        
        return {
            'negative_score': negative_count + (3 if pain_love_found else 0),  # Increased weight
            'positive_score': positive_count + (3 if positive_love_found else 0),  # Increased weight
            'neutral_score': neutral_count,
            'pain_love': pain_love_found,
            'positive_love': positive_love_found
        }
    
    def correct_emotion(self, text, current_emotion):
        """Correct the emotion label based on linguistic analysis"""
        analysis = self.analyze_emotion_content(text)
        
        # Strong negative indicators
        if analysis['negative_score'] >= 2 or analysis['pain_love']:
            return 'negative'
        
        # Strong positive indicators  
        elif analysis['positive_score'] >= 2 or analysis['positive_love']:
            return 'positive'
        
        # Mild sentiment
        elif analysis['negative_score'] > analysis['positive_score']:
            return 'negative'
        elif analysis['positive_score'] > analysis['negative_score']:
            return 'positive'
        
        # Default to neutral for ambiguous cases
        else:
            return 'neutral'

def fix_original_dataset():
    """Fix emotion labels on the original full dataset"""
    print("🔧 FIXING ORIGINAL DATASET LABELS (1000 SAMPLES)")
    print("=" * 70)
    
    # Load the original dataset with emotions
    df = pd.read_excel('datasets/output_with_emotions.xlsx')
    print(f"📂 Loaded original dataset: {df.shape[0]} samples")
    
    # Initialize corrector
    corrector = HindiUrduEmotionCorrector()
    
    # Analyze current labels
    print(f"\n📊 ORIGINAL EMOTION DISTRIBUTION:")
    current_dist = df['emotion'].value_counts()
    for emotion, count in current_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {emotion}: {count} samples ({percentage:.1f}%)")
    
    # Correct emotions
    print(f"\n🔧 CORRECTING EMOTION LABELS...")
    corrected_emotions = []
    changes = defaultdict(int)
    
    for i, row in df.iterrows():
        original_emotion = row['emotion']
        corrected_emotion = corrector.correct_emotion(row['text'], original_emotion)
        corrected_emotions.append(corrected_emotion)
        
        if original_emotion != corrected_emotion:
            changes[f"{original_emotion} -> {corrected_emotion}"] += 1
    
    # Update dataframe
    df['emotion_corrected'] = corrected_emotions
    
    print(f"\n📈 CORRECTION SUMMARY:")
    total_changes = sum(changes.values())
    print(f"  Total changes: {total_changes}/{len(df)} ({total_changes/len(df)*100:.1f}%)")
    for change, count in changes.items():
        print(f"  {change}: {count} changes")
    
    # Show new distribution
    print(f"\n📊 CORRECTED EMOTION DISTRIBUTION:")
    new_dist = df['emotion_corrected'].value_counts()
    for emotion, count in new_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {emotion}: {count} samples ({percentage:.1f}%)")
    
    # Save corrected full dataset
    output_file = 'datasets/output_with_emotions_corrected_full.xlsx'
    df_corrected = df[['text', 'emotion_corrected']].copy()
    df_corrected.columns = ['text', 'emotion']
    df_corrected.to_excel(output_file, index=False)
    
    print(f"\n💾 SAVED CORRECTED FULL DATASET:")
    print(f"  File: {output_file}")
    print(f"  Shape: {df_corrected.shape}")
    
    return df_corrected, changes

def balance_corrected_dataset(df_corrected):
    """Balance the corrected dataset properly"""
    print(f"\n\n🔄 BALANCING CORRECTED DATASET")
    print("=" * 70)
    
    # Get emotion counts
    emotion_counts = df_corrected['emotion'].value_counts()
    print(f"📊 CORRECTED DISTRIBUTION BEFORE BALANCING:")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(df_corrected)) * 100
        print(f"  {emotion}: {count} samples ({percentage:.1f}%)")
    
    # Find minimum class size for balancing
    min_samples = emotion_counts.min()
    print(f"\n⚖️ BALANCING STRATEGY:")
    print(f"  Minimum class size: {min_samples}")
    print(f"  Balancing method: Undersample to {min_samples} per class")
    
    # Balance by undersampling
    balanced_dfs = []
    for emotion in emotion_counts.index:
        emotion_df = df_corrected[df_corrected['emotion'] == emotion]
        if len(emotion_df) > min_samples:
            # Undersample
            emotion_balanced = emotion_df.sample(n=min_samples, random_state=42)
            print(f"  {emotion}: {len(emotion_df)} -> {len(emotion_balanced)} (undersampled)")
        else:
            emotion_balanced = emotion_df
            print(f"  {emotion}: {len(emotion_df)} -> {len(emotion_balanced)} (no change)")
        
        balanced_dfs.append(emotion_balanced)
    
    # Combine balanced data
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 FINAL BALANCED DISTRIBUTION:")
    final_dist = df_balanced['emotion'].value_counts()
    for emotion, count in final_dist.items():
        percentage = (count / len(df_balanced)) * 100
        print(f"  {emotion}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n📈 BALANCING SUMMARY:")
    print(f"  Original size: {len(df_corrected)} samples")
    print(f"  Balanced size: {len(df_balanced)} samples")
    print(f"  Reduction: {len(df_corrected) - len(df_balanced)} samples")
    
    # Save balanced dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'datasets/output_with_emotions_corrected_balanced_{timestamp}.xlsx'
    df_balanced.to_excel(output_file, index=False)
    
    print(f"\n💾 SAVED CORRECTED & BALANCED DATASET:")
    print(f"  File: {output_file}")
    print(f"  Shape: {df_balanced.shape}")
    print(f"  Ready for model training!")
    
    return df_balanced, output_file

def show_correction_examples(df_original, df_corrected):
    """Show examples of the corrections made"""
    print(f"\n\n📝 EXAMPLES OF MAJOR CORRECTIONS:")
    print("=" * 70)
    
    # Find samples where corrections were made
    df_with_original = df_original.copy()
    df_with_original['emotion_corrected'] = df_corrected['emotion']
    
    corrected_samples = df_with_original[df_with_original['emotion'] != df_with_original['emotion_corrected']]
    
    # Show examples by correction type
    correction_types = corrected_samples.groupby(['emotion', 'emotion_corrected']).size().sort_values(ascending=False)
    
    for (orig, corr), count in correction_types.head(5).items():
        print(f"\n🔄 {orig.upper()} -> {corr.upper()} ({count} corrections):")
        examples = corrected_samples[(corrected_samples['emotion'] == orig) & 
                                   (corrected_samples['emotion_corrected'] == corr)].head(3)
        
        for i, row in examples.iterrows():
            print(f"  📄 {row['text'][:120]}...")
            
            # Show analysis
            corrector = HindiUrduEmotionCorrector()
            analysis = corrector.analyze_emotion_content(row['text'])
            print(f"     Analysis: neg={analysis['negative_score']}, pos={analysis['positive_score']}, neu={analysis['neutral_score']}")
            if analysis['pain_love']:
                print(f"     💔 Contains pain+love pattern")
            if analysis['positive_love']:
                print(f"     💕 Contains positive+love pattern")
            print()

def main():
    """Main function to fix and balance the dataset"""
    print("🚀 EMOTION DATASET CORRECTION & BALANCING PIPELINE")
    print("=" * 70)
    
    # Step 1: Fix original dataset labels
    df_corrected_full, correction_stats = fix_original_dataset()
    
    # Step 2: Balance the corrected dataset
    df_balanced, balanced_file = balance_corrected_dataset(df_corrected_full)
    
    # Step 3: Show correction examples
    df_original = pd.read_excel('datasets/output_with_emotions.xlsx')
    show_correction_examples(df_original, df_corrected_full)
    
    print(f"\n✅ PIPELINE COMPLETE!")
    print(f"📁 Corrected full dataset: datasets/output_with_emotions_corrected_full.xlsx")
    print(f"📁 Corrected balanced dataset: {balanced_file}")
    print(f"🎯 Ready to retrain all models with proper labels!")
    
    return balanced_file

if __name__ == "__main__":
    balanced_dataset_file = main() 