#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioBERT + BIO Embeddings Runner Script

Simple script to run BioBERT with BIO word embeddings emotion classification.
"""

import sys
import os
import subprocess
from datetime import datetime

def main():
    print("🚀 BIOBERT + BIO EMBEDDINGS EMOTION CLASSIFICATION RUNNER")
    print("=" * 60)
    
    # Get the balanced dataset file
    balanced_file = "../../datasets/output_with_emotions_undersample.xlsx"
    
    if not os.path.exists(balanced_file):
        print("❌ Balanced dataset not found!")
        print(f"Looking for: {balanced_file}")
        print("Please make sure the balanced dataset exists.")
        return
    
    print(f"📂 Using dataset: {balanced_file}")
    
    # Run the BioBERT + BIO classifier
    try:
        print("\n🧬➕ Starting BioBERT + BIO embeddings training...")
        
        # Create results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Run the training script
        cmd = [
            "python", "biobert_bio_emotion_classifier.py",
            "--data", balanced_file,
            "--epochs", "3",
            "--batch_size", "8",  # Smaller batch size for stability
            "--learning_rate", "2e-5",
            "--max_length", "512",
            "--save_dir", results_dir
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ BioBERT + BIO embeddings training completed successfully!")
            print(result.stdout)
        else:
            print("❌ BioBERT + BIO embeddings training failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"❌ Error running BioBERT + BIO embeddings: {str(e)}")

if __name__ == "__main__":
    main() 