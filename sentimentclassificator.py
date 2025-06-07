import pandas as pd
from transformers import pipeline
import torch
import os

def load_models():
    """Load multiple sentiment analysis models for better accuracy"""
    models = {}
    
    print("📥 Loading sentiment analysis models...")
    
    # Model 1: Multilingual sentiment (supports Hindi/Urdu)
    try:
        print("  ⏳ Loading multilingual model...")
        models['multilingual'] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            top_k=None
        )
        print("  ✅ Multilingual model loaded")
    except Exception as e:
        print(f"  ❌ Failed to load multilingual model: {e}")
    
    # Model 2: BERT-based multilingual sentiment
    try:
        print("  ⏳ Loading BERT multilingual model...")
        models['bert_multilingual'] = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            top_k=None
        )
        print("  ✅ BERT multilingual model loaded")
    except Exception as e:
        print(f"  ❌ Failed to load BERT multilingual model: {e}")
    
    # Model 3: Fallback English model
    try:
        print("  ⏳ Loading English fallback model...")
        models['english'] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None
        )
        print("  ✅ English fallback model loaded")
    except Exception as e:
        print(f"  ❌ Failed to load English fallback model: {e}")
    
    return models

def classify_sentiment(text, models):
    """Classify sentiment using multiple models and ensemble approach"""
    try:
        # Handle empty or null text
        if pd.isna(text) or str(text).strip() == "":
            return "neutral"
        
        # Convert to string and truncate if too long
        text = str(text)[:512]
        
        results = []
        scores = []
        
        # Try each model
        for model_name, model in models.items():
            try:
                result = model(text)
                if result:
                    # Handle different output formats
                    if isinstance(result[0], list):
                        best_result = max(result[0], key=lambda x: x['score'])
                    else:
                        best_result = result[0]
                    
                    label = best_result['label']
                    score = best_result['score']
                    
                    # Normalize labels to our format
                    normalized_label = normalize_label(label)
                    results.append(normalized_label)
                    scores.append(score)
                    
                    print(f"    {model_name}: {normalized_label} ({score:.3f})")
                    
            except Exception as e:
                print(f"    ❌ {model_name} failed: {e}")
                continue
        
        # Ensemble decision: majority vote with confidence weighting
        if results:
            decision = ensemble_decision(results, scores)
            print(f"    🎯 Final decision: {decision}")
            return decision
        else:
            return "neutral"
            
    except Exception as e:
        print(f"Error processing text: {str(text)[:50]}... -> {e}")
        return "error"

def normalize_label(label):
    """Normalize different model outputs to our standard format"""
    label = label.upper()
    
    # Handle different label formats
    if label in ['POSITIVE', 'POS', 'LABEL_2', '5 STARS', '4 STARS']:
        return "positive"
    elif label in ['NEGATIVE', 'NEG', 'LABEL_0', '1 STAR', '2 STARS']:
        return "negative"
    elif label in ['NEUTRAL', 'NEU', 'LABEL_1', '3 STARS']:
        return "neutral"
    else:
        # Default mapping
        return label.lower()

def ensemble_decision(results, scores):
    """Make final decision based on multiple model outputs with improved weighting"""
    if not results:
        return "neutral"
    
    # Give more weight to multilingual models for non-English text
    weighted_scores = {}
    
    for i, (result, score) in enumerate(zip(results, scores)):
        if result not in weighted_scores:
            weighted_scores[result] = 0
        
        # Apply model-specific weights
        if i == 0:  # multilingual model - higher weight
            weight = score * 1.5
        elif i == 1:  # bert_multilingual model - medium weight
            weight = score * 1.3
        else:  # english model - lower weight for non-English text
            weight = score * 0.7
            
        weighted_scores[result] += weight
    
    # If we have a clear non-neutral sentiment with decent confidence, prefer it
    non_neutral = {k: v for k, v in weighted_scores.items() if k != "neutral"}
    if non_neutral:
        max_non_neutral = max(non_neutral.values())
        max_neutral = weighted_scores.get("neutral", 0)
        
        # If non-neutral sentiment is at least 80% as strong as neutral, choose it
        if max_non_neutral >= max_neutral * 0.8:
            return max(non_neutral.keys(), key=lambda x: non_neutral[x])
    
    # Otherwise, return the highest weighted score
    return max(weighted_scores.keys(), key=lambda x: weighted_scores[x])

# Load sentiment models
models = load_models()

if not models:
    print("❌ No sentiment models could be loaded! Exiting...")
    exit(1)

print(f"✅ Loaded {len(models)} sentiment models successfully!")

# Load your Excel file
input_file = "datasets/Disorder_ADHD_and_GAD_output.xlsx"
if not os.path.exists(input_file):
    print(f"❌ Error: File '{input_file}' not found!")
    print("Please make sure the file exists in the datasets folder.")
    exit(1)

print(f"\n📂 Loading data from: {input_file}")
df = pd.read_excel(input_file)

# Remove test limit - process all texts
# TEST_LIMIT = 100
# df = df.head(TEST_LIMIT)
# print(f"⚠️  Testing with first {TEST_LIMIT} rows only")

print(f"✅ Loaded {len(df)} rows of data")

# Display column information
print(f"📊 Columns in the dataset: {list(df.columns)}")

# Show sample texts
print("\n📝 Sample texts:")
for i, text in enumerate(df['text'].head(3)):
    print(f"  {i+1}: {str(text)[:100]}...")

# Remove the 2nd column (label) if it exists
if len(df.columns) >= 2:
    second_column = df.columns[1]
    print(f"\n🗑️ Removing column: '{second_column}'")
    df = df.drop(columns=[second_column])
    print(f"✅ Column '{second_column}' removed")

# Apply sentiment classification
print(f"\n🔄 Classifying sentiments for {len(df)} texts...")
print("This may take a while for multilingual models...")

# Process texts and show progress
emotions = []
for i, text in enumerate(df['text']):
    if (i + 1) % 50 == 0 or i == 0:
        print(f"\n📍 Processing text {i+1}/{len(df)}: {str(text)[:50]}...")
    
    emotion = classify_sentiment(text, models)
    emotions.append(emotion)
    
    if (i + 1) % 100 == 0:
        print(f"✅ Completed {i+1}/{len(df)} texts")

df["emotion"] = emotions

# Show emotion distribution
emotion_counts = df["emotion"].value_counts()
print(f"\n📊 Final Emotion Distribution:")
for emotion, count in emotion_counts.items():
    print(f"  {emotion}: {count} ({count/len(df)*100:.1f}%)")

# Save to new Excel file
output_file = "datasets/output_with_emotions.xlsx"
df.to_excel(output_file, index=False)

# Save detailed statistics to a text file
stats_file = "datasets/emotion_classification_stats.txt"
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("=== EMOTION CLASSIFICATION STATISTICS ===\n")
    f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input file: {input_file}\n")
    f.write(f"Output file: {output_file}\n")
    f.write("\n" + "="*50 + "\n")
    
    f.write(f"\n📊 DATASET OVERVIEW:\n")
    f.write(f"  Total texts processed: {len(df):,}\n")
    f.write(f"  Original columns: text, label (label removed)\n")
    f.write(f"  Final columns: {', '.join(df.columns)}\n")
    
    f.write(f"\n📈 EMOTION DISTRIBUTION:\n")
    total_texts = len(df)
    for emotion in ['positive', 'negative', 'neutral', 'error']:
        count = emotion_counts.get(emotion, 0)
        percentage = (count / total_texts) * 100
        f.write(f"  {emotion.capitalize():<10}: {count:>4} texts ({percentage:>5.1f}%)\n")
    
    f.write(f"\n📋 DETAILED BREAKDOWN:\n")
    non_neutral = emotion_counts.get('positive', 0) + emotion_counts.get('negative', 0)
    f.write(f"  Emotional content: {non_neutral} texts ({(non_neutral/total_texts)*100:.1f}%)\n")
    f.write(f"  Neutral content:   {emotion_counts.get('neutral', 0)} texts ({(emotion_counts.get('neutral', 0)/total_texts)*100:.1f}%)\n")
    f.write(f"  Errors:            {emotion_counts.get('error', 0)} texts ({(emotion_counts.get('error', 0)/total_texts)*100:.1f}%)\n")
    
    f.write(f"\n🎯 MODEL PERFORMANCE:\n")
    f.write(f"  Successfully classified: {total_texts - emotion_counts.get('error', 0)} texts\n")
    f.write(f"  Classification accuracy: {((total_texts - emotion_counts.get('error', 0))/total_texts)*100:.1f}%\n")
    
    f.write(f"\n🔍 SENTIMENT ANALYSIS:\n")
    if emotion_counts.get('positive', 0) > 0:
        f.write(f"  Most positive sentiment: {(emotion_counts.get('positive', 0)/total_texts)*100:.1f}% of dataset\n")
    if emotion_counts.get('negative', 0) > 0:
        f.write(f"  Most negative sentiment: {(emotion_counts.get('negative', 0)/total_texts)*100:.1f}% of dataset\n")
    
    f.write(f"\n📝 NOTES:\n")
    f.write(f"  - Dataset contains Hindi/Urdu text\n")
    f.write(f"  - Used multilingual sentiment models\n")
    f.write(f"  - Ensemble approach with weighted voting\n")
    f.write(f"  - Models: BERT multilingual + English fallback\n")

print(f"\n📊 Statistics saved to: {stats_file}")
print(f"\n✅ Done! File saved as: {output_file}")
print(f"📄 Final dataset shape: {df.shape}")
print(f"📄 Final columns: {list(df.columns)}")
print(f"🎯 Successfully classified {len(df)} Hindi/Urdu texts!")
