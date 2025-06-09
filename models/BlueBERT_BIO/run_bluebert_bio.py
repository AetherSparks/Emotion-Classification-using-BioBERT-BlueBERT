#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlueBERT + BIO Embeddings Emotion Classification Runner

This script runs the BlueBERT + BIO word embeddings model training and evaluation.
"""

import subprocess
import sys
import os

def run_bluebert_bio_training():
    """Run BlueBERT + BIO model training"""
    print("🔵🧬 Starting BlueBERT + BIO Embeddings Training...")
    print("=" * 60)
    
    # Define the command
    cmd = [
        sys.executable, 
        "bluebert_bio_emotion_classifier.py",
        "--data", "../../datasets/output_with_emotions_undersample.xlsx",
        "--epochs", "3",
        "--batch_size", "8",
        "--learning_rate", "2e-5",
        "--max_length", "512",
        "--save_dir", "results"
    ]
    
    print("Command:", " ".join(cmd))
    print("-" * 60)
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ BlueBERT + BIO training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ BlueBERT + BIO training failed with error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = run_bluebert_bio_training()
    if success:
        print("\n🎉 BlueBERT + BIO Model Training Complete!")
    else:
        print("\n💥 BlueBERT + BIO Model Training Failed!")
        sys.exit(1) 