import pandas as pd

# Analyze original dataset
print("=== ORIGINAL DATASET ANALYSIS ===")
df = pd.read_excel('datasets/output_with_emotions.xlsx')
print(f"Original dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nEmotion distribution:")
print(df['emotion'].value_counts())
print("\nEmotion percentages:")
print(df['emotion'].value_counts(normalize=True) * 100)

print("\nFirst 10 texts and their emotions:")
for i in range(10):
    text = str(df.iloc[i]['text'])[:150]
    emotion = df.iloc[i]['emotion']
    print(f"{i+1:2d}. [{emotion:8s}] {text}...")

# Analyze balanced dataset
print("\n\n=== BALANCED DATASET ANALYSIS ===")
df_balanced = pd.read_excel('datasets/output_with_emotions_undersample.xlsx')
print(f"Balanced dataset shape: {df_balanced.shape}")
print(f"Columns: {df_balanced.columns.tolist()}")
print("\nEmotion distribution:")
print(df_balanced['emotion'].value_counts())

print("\nFirst 10 texts and their emotions from balanced:")
for i in range(10):
    text = str(df_balanced.iloc[i]['text'])[:150]
    emotion = df_balanced.iloc[i]['emotion']
    print(f"{i+1:2d}. [{emotion:8s}] {text}...")

# Check for data quality issues
print("\n\n=== DATA QUALITY ANALYSIS ===")
print("Text length analysis:")
df_balanced['text_length'] = df_balanced['text'].str.len()
print(f"Average text length: {df_balanced['text_length'].mean():.1f}")
print(f"Min text length: {df_balanced['text_length'].min()}")
print(f"Max text length: {df_balanced['text_length'].max()}")

print("\nTexts shorter than 20 characters:")
short_texts = df_balanced[df_balanced['text_length'] < 20]
print(f"Count: {len(short_texts)}")
for i, row in short_texts.head(5).iterrows():
    print(f"  '{row['text']}' -> {row['emotion']}")

print("\nTexts longer than 500 characters:")
long_texts = df_balanced[df_balanced['text_length'] > 500]
print(f"Count: {len(long_texts)}")
for i, row in long_texts.head(3).iterrows():
    print(f"  '{row['text'][:100]}...' -> {row['emotion']}")

# Check if labels make sense
print("\n\n=== LABEL QUALITY CHECK ===")
print("Sample positive emotion texts:")
positive_samples = df_balanced[df_balanced['emotion'] == 'positive'].head(5)
for i, row in positive_samples.iterrows():
    print(f"  [{row['emotion']}] {row['text'][:150]}...")

print("\nSample negative emotion texts:")
negative_samples = df_balanced[df_balanced['emotion'] == 'negative'].head(5)
for i, row in negative_samples.iterrows():
    print(f"  [{row['emotion']}] {row['text'][:150]}...")

print("\nSample neutral emotion texts:")
neutral_samples = df_balanced[df_balanced['emotion'] == 'neutral'].head(5)
for i, row in neutral_samples.iterrows():
    print(f"  [{row['emotion']}] {row['text'][:150]}...") 