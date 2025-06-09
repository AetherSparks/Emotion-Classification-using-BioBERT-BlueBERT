# BlueBERT + BIO Word Embeddings Emotion Classification

This model combines **BlueBERT** (a biomedical BERT model) with custom **BIO word embeddings** for enhanced emotion classification in clinical and biomedical texts.

## Model Architecture

### Base Model

- **BlueBERT**: `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12`
- Pre-trained on PubMed abstracts and MIMIC-III clinical notes
- 12 layers, 768 hidden units, 12 attention heads

### Enhanced Features

- **Custom BIO Embeddings**: 60+ biomedical and clinical terms (100D embeddings)
- **Clinical Fusion Layer**: Combines BlueBERT (768D) + BIO (100D) features
- **Clinical Attention**: Multi-head attention mechanism for clinical context
- **Enhanced Classification**: Multi-layer classifier with normalization

### BIO Terms Categories

- Medical conditions (disease, disorder, cancer, etc.)
- Anatomy (heart, brain, lung, etc.)
- Symptoms (pain, fever, fatigue, etc.)
- Medications (drug, therapy, treatment, etc.)
- Clinical context (patient, doctor, hospital, etc.)
- Emotional/psychological terms (stress, mood, anxiety, etc.)

## Usage

### Training

```bash
python run_bluebert_bio.py
```

### Direct Script Usage

```bash
python bluebert_bio_emotion_classifier.py \
    --data ../../datasets/output_with_emotions_undersample.xlsx \
    --epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --max_length 512 \
    --save_dir results
```

## Model Parameters

- **Total Parameters**: ~108.7M
- **BlueBERT Parameters**: ~108.3M
- **BIO Embedding Parameters**: ~25K
- **Fusion Parameters**: ~200K
- **Attention Parameters**: ~1M

## Expected Performance

The model is designed for 3-class emotion classification:

- **Negative** emotion
- **Neutral** emotion
- **Positive** emotion

Expected metrics on balanced dataset (246 samples):

- Accuracy: ~35-40%
- F1-Score (Macro): ~25-30%
- Clinical term enhancement over base BlueBERT

## Output Files

After training, the following files are generated:

- `bluebert_bio_metrics_[timestamp].json` - Detailed metrics
- `bluebert_bio_report_[timestamp].txt` - Human-readable report
- `bluebert_bio_model_[timestamp].pth` - Trained model
- `confusion_matrix_[timestamp].png` - Confusion matrix visualization
- `training_history_[timestamp].png` - Training progress plots

## Features

- ✅ Enhanced clinical text understanding
- ✅ BIO word embeddings integration
- ✅ Clinical attention mechanism
- ✅ Comprehensive evaluation metrics
- ✅ Proper epoch visualization (fixed)
- ✅ Matthews Correlation Coefficient (MCC)
- ✅ Multi-class AUC-ROC
- ✅ Per-emotion detailed analysis
- ✅ Clinical-focused model architecture
