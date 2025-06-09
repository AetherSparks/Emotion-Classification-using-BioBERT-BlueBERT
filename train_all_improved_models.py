#!/usr/bin/env python3
"""
Train All Improved Models - Enhanced Performance Script
Run all models with improved hyperparameters and expanded Hindi vocabulary
"""

import os
import subprocess
import time
from datetime import datetime

def run_command(command, model_name):
    """Run a command and track time"""
    print(f"\n🚀 Starting {model_name} training...")
    print(f"Command: {command}")
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ {model_name} completed in {duration/60:.1f} minutes")
        return True, duration
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ {model_name} failed after {duration/60:.1f} minutes")
        print(f"Error: {e.stderr}")
        return False, duration

def main():
    print("🎯 TRAINING ALL IMPROVED MODELS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📋 IMPROVEMENTS APPLIED:")
    print("✅ Epochs: 5 → 10 (double training)")
    print("✅ Hindi vocabulary: 146 → 200+ terms")
    print("✅ Enhanced emotional categories")
    print("✅ Poetry & literature terms added")
    
    # Model training commands
    models = [
        {
            "name": "MultiBERT (Basic)",
            "command": "cd models/MultiBERT && python multibert_emotion_classifier.py --epochs 10",
            "expected_improvement": "53% → 58%+"
        },
        {
            "name": "MultiBERT + Hindi Features",
            "command": "cd models/MultiBERT && python multibert_emotion_classifier.py --epochs 10 --use_hindi_features",
            "expected_improvement": "49% → 56%+"
        },
        {
            "name": "BioBERT + Enhanced Hindi",
            "command": "cd models/BioBERT_BIO && python biobert_bio_emotion_classifier.py --epochs 10",
            "expected_improvement": "44% → 52%+"
        },
        {
            "name": "BlueBERT + Enhanced Hindi",
            "command": "cd models/BlueBERT_BIO && python bluebert_bio_emotion_classifier.py --epochs 10",
            "expected_improvement": "52% → 60%+"
        }
    ]
    
    results = []
    total_start = time.time()
    
    for i, model in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"📊 MODEL {i}/{len(models)}: {model['name']}")
        print(f"🎯 Expected: {model['expected_improvement']}")
        print(f"{'='*60}")
        
        success, duration = run_command(model['command'], model['name'])
        results.append({
            'name': model['name'],
            'success': success,
            'duration': duration,
            'expected': model['expected_improvement']
        })
    
    total_duration = time.time() - total_start
    
    # Print summary
    print(f"\n{'='*60}")
    print("📋 TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_duration/60:.1f} minutes")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📊 RESULTS:")
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"  {result['name']:<30} {status:<10} {result['duration']/60:>6.1f}m  {result['expected']}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"\n🎯 SUCCESS RATE: {successful}/{len(results)} models trained successfully")
    
    if successful > 0:
        print(f"\n📈 NEXT STEPS:")
        print(f"1. Check results in models/*/results/ directories")
        print(f"2. Run comparison: cd models/MultiBERT && python multibert_comparison_report.py")
        print(f"3. View improvements in accuracy and F1 scores")

if __name__ == "__main__":
    main() 