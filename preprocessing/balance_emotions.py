#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion Dataset Balancing Script

This script balances the emotion classes in the output_with_emotions.xlsx file
by applying undersampling or oversampling techniques to create a balanced dataset.

Author: Data Preprocessing Pipeline
Date: Generated for Emotion Classification Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class EmotionBalancer:
    def __init__(self, input_file, output_folder):
        """
        Initialize the emotion balancer
        
        Args:
            input_file (str): Path to the input Excel file
            output_folder (str): Path to the output folder
        """
        self.input_file = Path(input_file)
        self.output_folder = Path(output_folder)
        self.df = None
        self.balanced_df = None
        
    def load_data(self):
        """Load the emotion data from Excel file"""
        try:
            print(f"📂 Loading data from: {self.input_file}")
            self.df = pd.read_excel(self.input_file)
            
            # Ensure we have the required columns
            if 'emotion' not in self.df.columns:
                raise ValueError("Dataset must contain 'emotion' column")
            if 'text' not in self.df.columns:
                raise ValueError("Dataset must contain 'text' column")
                
            print(f"✅ Successfully loaded {len(self.df)} records")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def analyze_distribution(self):
        """Analyze the current emotion distribution"""
        if self.df is None:
            print("❌ No data loaded!")
            return
            
        print("\n📊 CURRENT EMOTION DISTRIBUTION:")
        print("=" * 50)
        
        emotion_counts = self.df['emotion'].value_counts()
        total_samples = len(self.df)
        
        for emotion, count in emotion_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {emotion:<10}: {count:>6} texts ({percentage:>5.1f}%)")
        
        print(f"\n  Total     : {total_samples:>6} texts")
        
        # Calculate imbalance ratio
        max_class = emotion_counts.max()
        min_class = emotion_counts.min()
        imbalance_ratio = max_class / min_class
        
        print(f"\n🔍 IMBALANCE ANALYSIS:")
        print(f"  Majority class: {emotion_counts.idxmax()} with {max_class} samples")
        print(f"  Minority class: {emotion_counts.idxmin()} with {min_class} samples")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        return emotion_counts
    
    def undersample_balance(self):
        """Balance the dataset using undersampling (reduce majority classes)"""
        if self.df is None:
            print("❌ No data loaded!")
            return False
            
        print("\n🔧 APPLYING UNDERSAMPLING BALANCE...")
        
        # Get current distribution
        emotion_counts = self.df['emotion'].value_counts()
        min_samples = emotion_counts.min()
        
        print(f"📉 Target samples per class: {min_samples}")
        
        # Undersample each class to match the minority class
        balanced_data = []
        
        for emotion in emotion_counts.index:
            emotion_data = self.df[self.df['emotion'] == emotion]
            
            if len(emotion_data) > min_samples:
                # Undersample
                undersampled = resample(emotion_data, 
                                      replace=False, 
                                      n_samples=min_samples, 
                                      random_state=42)
                balanced_data.append(undersampled)
                print(f"  {emotion}: {len(emotion_data)} → {len(undersampled)} samples")
            else:
                # Keep all samples if already at minimum
                balanced_data.append(emotion_data)
                print(f"  {emotion}: {len(emotion_data)} samples (unchanged)")
        
        # Combine all balanced data
        self.balanced_df = pd.concat(balanced_data, ignore_index=True)
        
        # Shuffle the dataset
        self.balanced_df = self.balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Undersampling complete! Total samples: {len(self.balanced_df)}")
        return True
    
    def oversample_balance(self):
        """Balance the dataset using oversampling (increase minority classes)"""
        if self.df is None:
            print("❌ No data loaded!")
            return False
            
        print("\n🔧 APPLYING OVERSAMPLING BALANCE...")
        
        # Get current distribution
        emotion_counts = self.df['emotion'].value_counts()
        max_samples = emotion_counts.max()
        
        print(f"📈 Target samples per class: {max_samples}")
        
        # Oversample each class to match the majority class
        balanced_data = []
        
        for emotion in emotion_counts.index:
            emotion_data = self.df[self.df['emotion'] == emotion]
            
            if len(emotion_data) < max_samples:
                # Oversample
                oversampled = resample(emotion_data, 
                                     replace=True, 
                                     n_samples=max_samples, 
                                     random_state=42)
                balanced_data.append(oversampled)
                print(f"  {emotion}: {len(emotion_data)} → {len(oversampled)} samples")
            else:
                # Keep all samples if already at maximum
                balanced_data.append(emotion_data)
                print(f"  {emotion}: {len(emotion_data)} samples (unchanged)")
        
        # Combine all balanced data
        self.balanced_df = pd.concat(balanced_data, ignore_index=True)
        
        # Shuffle the dataset
        self.balanced_df = self.balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Oversampling complete! Total samples: {len(self.balanced_df)}")
        return True
    
    def custom_balance(self, target_samples):
        """Balance the dataset to a custom number of samples per class"""
        if self.df is None:
            print("❌ No data loaded!")
            return False
            
        print(f"\n🔧 APPLYING CUSTOM BALANCE (Target: {target_samples} per class)...")
        
        # Get current distribution
        emotion_counts = self.df['emotion'].value_counts()
        
        # Balance each class to target samples
        balanced_data = []
        
        for emotion in emotion_counts.index:
            emotion_data = self.df[self.df['emotion'] == emotion]
            current_samples = len(emotion_data)
            
            if current_samples == target_samples:
                # Already at target
                balanced_data.append(emotion_data)
                print(f"  {emotion}: {current_samples} samples (unchanged)")
            elif current_samples > target_samples:
                # Undersample
                resampled = resample(emotion_data, 
                                   replace=False, 
                                   n_samples=target_samples, 
                                   random_state=42)
                balanced_data.append(resampled)
                print(f"  {emotion}: {current_samples} → {target_samples} samples (undersampled)")
            else:
                # Oversample
                resampled = resample(emotion_data, 
                                   replace=True, 
                                   n_samples=target_samples, 
                                   random_state=42)
                balanced_data.append(resampled)
                print(f"  {emotion}: {current_samples} → {target_samples} samples (oversampled)")
        
        # Combine all balanced data
        self.balanced_df = pd.concat(balanced_data, ignore_index=True)
        
        # Shuffle the dataset
        self.balanced_df = self.balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Custom balancing complete! Total samples: {len(self.balanced_df)}")
        return True
    
    def save_balanced_data(self, method="balanced"):
        """Save the balanced dataset to a new file"""
        if self.balanced_df is None:
            print("❌ No balanced data to save!")
            return False
            
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_folder / f"output_with_emotions_{method}_{timestamp}.xlsx"
        
        try:
            print(f"\n💾 Saving balanced dataset to: {output_file}")
            self.balanced_df.to_excel(output_file, index=False)
            
            print(f"✅ Successfully saved {len(self.balanced_df)} records")
            
            # Generate statistics report
            self.generate_balance_report(output_file, method)
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")
            return False
    
    def generate_balance_report(self, output_file, method):
        """Generate a detailed report of the balancing process"""
        report_file = self.output_folder / f"balance_report_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== EMOTION BALANCING REPORT ===\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Output file: {output_file}\n")
            f.write(f"Balancing method: {method.upper()}\n\n")
            
            f.write("=" * 50 + "\n\n")
            
            # Original distribution
            f.write("📊 ORIGINAL EMOTION DISTRIBUTION:\n")
            original_counts = self.df['emotion'].value_counts()
            original_total = len(self.df)
            
            for emotion, count in original_counts.items():
                percentage = (count / original_total) * 100
                f.write(f"  {emotion:<10}: {count:>6} texts ({percentage:>5.1f}%)\n")
            f.write(f"  Total     : {original_total:>6} texts\n\n")
            
            # Balanced distribution
            f.write("📊 BALANCED EMOTION DISTRIBUTION:\n")
            balanced_counts = self.balanced_df['emotion'].value_counts()
            balanced_total = len(self.balanced_df)
            
            for emotion, count in balanced_counts.items():
                percentage = (count / balanced_total) * 100
                f.write(f"  {emotion:<10}: {count:>6} texts ({percentage:>5.1f}%)\n")
            f.write(f"  Total     : {balanced_total:>6} texts\n\n")
            
            # Statistics
            f.write("📈 BALANCING STATISTICS:\n")
            f.write(f"  Original dataset size: {original_total}\n")
            f.write(f"  Balanced dataset size: {balanced_total}\n")
            f.write(f"  Size change: {balanced_total - original_total:+d} samples\n")
            f.write(f"  Size ratio: {balanced_total/original_total:.2f}x\n\n")
            
            f.write("🎯 BALANCE QUALITY:\n")
            balance_std = balanced_counts.std()
            f.write(f"  Standard deviation: {balance_std:.2f}\n")
            f.write(f"  Balance quality: {'Perfect' if balance_std == 0 else 'Good' if balance_std < 10 else 'Moderate'}\n")
        
        print(f"📊 Balance report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Balance emotion dataset')
    parser.add_argument('--input', default='datasets/output_with_emotions.xlsx', 
                       help='Input Excel file path')
    parser.add_argument('--output', default='datasets/', 
                       help='Output folder path')
    parser.add_argument('--method', choices=['undersample', 'oversample', 'custom'], 
                       default='undersample', help='Balancing method')
    parser.add_argument('--target', type=int, default=None, 
                       help='Target samples per class (for custom method)')
    
    args = parser.parse_args()
    
    print("🎯 EMOTION DATASET BALANCER")
    print("=" * 50)
    
    # Initialize balancer
    balancer = EmotionBalancer(args.input, args.output)
    
    # Load data
    if not balancer.load_data():
        return
    
    # Analyze current distribution
    balancer.analyze_distribution()
    
    # Apply balancing
    success = False
    if args.method == 'undersample':
        success = balancer.undersample_balance()
    elif args.method == 'oversample':
        success = balancer.oversample_balance()
    elif args.method == 'custom':
        if args.target is None:
            print("❌ Custom method requires --target parameter")
            return
        success = balancer.custom_balance(args.target)
    
    if not success:
        return
    
    # Show new distribution
    print("\n📊 NEW EMOTION DISTRIBUTION:")
    print("=" * 50)
    
    new_counts = balancer.balanced_df['emotion'].value_counts()
    new_total = len(balancer.balanced_df)
    
    for emotion, count in new_counts.items():
        percentage = (count / new_total) * 100
        print(f"  {emotion:<10}: {count:>6} texts ({percentage:>5.1f}%)")
    
    print(f"\n  Total     : {new_total:>6} texts")
    
    # Save balanced data
    balancer.save_balanced_data(args.method)
    
    print("\n🎉 BALANCING COMPLETE!")

if __name__ == "__main__":
    main() 