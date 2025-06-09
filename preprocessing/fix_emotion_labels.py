#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Emotion Labels for Hindi/Urdu Text

This script corrects the mislabeled emotions using proper Hindi/Urdu sentiment analysis
and manual rules for better emotion classification.
"""

import pandas as pd
import re
from collections import defaultdict

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
            'टूटा', 'बिखरा', 'शिकस्त', 'हार', 'निराशा', 'हताशा', 'बेकार'
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
            'उत्साह', 'जुनून', 'शौक', 'चाह', 'इच्छा', 'कामना', 'अरमान'
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
            r'दिल.*टूट', r'दिल.*दर्द', r'आंसू.*प्यार'
        ]
        
        self.positive_love_patterns = [
            r'खुश.*प्यार', r'प्यार.*खुशी', r'मुस्कुराना.*प्यार', r'हंसना.*इश्क़'
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
            'negative_score': negative_count + (2 if pain_love_found else 0),
            'positive_score': positive_count + (2 if positive_love_found else 0),
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

def fix_emotion_labels():
    """Fix the emotion labels in the balanced dataset"""
    print("🔧 FIXING EMOTION LABELS FOR HINDI/URDU TEXT")
    print("=" * 60)
    
    # Load the balanced dataset
    df = pd.read_excel('datasets/output_with_emotions_undersample.xlsx')
    print(f"📂 Loaded balanced dataset: {df.shape[0]} samples")
    
    # Initialize corrector
    corrector = HindiUrduEmotionCorrector()
    
    # Analyze current labels
    print(f"\n📊 CURRENT EMOTION DISTRIBUTION:")
    current_dist = df['emotion'].value_counts()
    for emotion, count in current_dist.items():
        print(f"  {emotion}: {count} samples")
    
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
        print(f"  {emotion}: {count} samples")
    
    # Show examples of corrections
    print(f"\n📝 EXAMPLES OF CORRECTIONS:")
    corrected_samples = df[df['emotion'] != df['emotion_corrected']].head(10)
    
    for i, row in corrected_samples.iterrows():
        print(f"\n{i+1}. Text: {row['text'][:100]}...")
        print(f"   Original: {row['emotion']} -> Corrected: {row['emotion_corrected']}")
        
        # Show why it was corrected
        analysis = corrector.analyze_emotion_content(row['text'])
        print(f"   Analysis: neg={analysis['negative_score']}, pos={analysis['positive_score']}, neu={analysis['neutral_score']}")
        if analysis['pain_love']:
            print(f"   Reason: Contains pain+love pattern")
    
    # Save corrected dataset
    output_file = 'datasets/output_with_emotions_corrected.xlsx'
    df_corrected = df[['text', 'emotion_corrected']].copy()
    df_corrected.columns = ['text', 'emotion']
    df_corrected.to_excel(output_file, index=False)
    
    print(f"\n💾 SAVED CORRECTED DATASET:")
    print(f"  File: {output_file}")
    print(f"  Shape: {df_corrected.shape}")
    print(f"  Final distribution:")
    final_dist = df_corrected['emotion'].value_counts()
    for emotion, count in final_dist.items():
        print(f"    {emotion}: {count} samples")
    
    return df_corrected, changes

if __name__ == "__main__":
    corrected_df, correction_stats = fix_emotion_labels()
    print(f"\n✅ EMOTION LABEL CORRECTION COMPLETE!")
    print(f"Ready to retrain models with corrected labels!") 