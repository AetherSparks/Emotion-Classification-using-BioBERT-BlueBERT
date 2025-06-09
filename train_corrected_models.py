#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train All Models on Corrected Dataset

This script trains all 4 models on the properly corrected and balanced dataset.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_model_training(model_name, model_dir, corrected_dataset):
    """Run training for a specific model"""
    print(f"\n🚀 TRAINING {model_name.upper()}")
    print("=" * 60)
    
    # Navigate to model directory
    original_dir = os.getcwd()
    model_path = os.path.join("models", model_dir)
    
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    try:
        os.chdir(model_path)
        
        # Define training script name
        if "BioBERT_BIO" in model_dir:
            script_name = "biobert_bio_emotion_classifier.py"
        elif "BlueBERT_BIO" in model_dir:
            script_name = "bluebert_bio_emotion_classifier.py"
        elif "BioBERT" in model_dir:
            script_name = "biobert_emotion_classifier.py"
        elif "BlueBERT" in model_dir:
            script_name = "bluebert_emotion_classifier.py"
        else:
            print(f"❌ Unknown model directory: {model_dir}")
            return False
        
        # Run training
        cmd = [
            sys.executable,
            script_name,
            "--data", f"../../{corrected_dataset}",
            "--epochs", "3",
            "--batch_size", "8",
            "--learning_rate", "2e-5",
            "--max_length", "512",
            "--save_dir", "results_corrected"
        ]
        
        print(f"📁 Directory: {os.getcwd()}")
        print(f"🔧 Command: {' '.join(cmd)}")
        print("-" * 60)
        
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ {model_name} training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {model_name} training failed with error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error in {model_name}: {e}")
        return False
    finally:
        os.chdir(original_dir)

def main():
    """Train all models on corrected dataset"""
    print("🎯 TRAINING ALL MODELS ON CORRECTED DATASET")
    print("=" * 70)
    
    # Find the corrected balanced dataset
    corrected_dataset = "datasets/output_with_emotions_corrected_balanced_20250609_140452.xlsx"
    
    if not os.path.exists(corrected_dataset):
        print(f"❌ Corrected dataset not found: {corrected_dataset}")
        print("Please run fix_original_labels.py first!")
        return
    
    print(f"📂 Using corrected dataset: {corrected_dataset}")
    
    # Define models to train
    models = [
        ("BioBERT", "BioBERT"),
        ("BlueBERT", "BlueBERT"),  
        ("BioBERT + BIO", "BioBERT_BIO"),
        ("BlueBERT + BIO", "BlueBERT_BIO")
    ]
    
    # Track results
    results = {}
    start_time = datetime.now()
    
    # Train each model
    for model_name, model_dir in models:
        model_start = datetime.now()
        success = run_model_training(model_name, model_dir, corrected_dataset)
        model_end = datetime.now()
        model_duration = model_end - model_start
        
        results[model_name] = {
            'success': success,
            'duration': model_duration
        }
        
        print(f"\n⏱️ {model_name} training duration: {model_duration}")
    
    # Print summary
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    print(f"\n\n📊 TRAINING SUMMARY")
    print("=" * 70)
    print(f"📅 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total duration: {total_duration}")
    print()
    
    successful_models = 0
    for model_name, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        duration = result['duration']
        print(f"  {model_name:<20}: {status} ({duration})")
        if result['success']:
            successful_models += 1
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"  Successful: {successful_models}/{len(models)} models")
    print(f"  Failed: {len(models) - successful_models}/{len(models)} models")
    
    if successful_models == len(models):
        print(f"\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print(f"📁 Results saved in: models/*/results_corrected/")
        print(f"🔍 Compare results to see the improvement!")
    else:
        print(f"\n⚠️ Some models failed. Check the error messages above.")

if __name__ == "__main__":
    main() 