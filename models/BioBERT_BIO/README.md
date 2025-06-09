# BioBERT + BIO Word Embeddings Emotion Classification Model

This directory contains an enhanced implementation of a BioBERT-based emotion classification system augmented with additional BIO (biomedical) word embeddings for improved biomedical text understanding.

## 🎯 Overview

The BioBERT + BIO emotion classifier combines the power of the pre-trained BioBERT model (`dmis-lab/biobert-base-cased-v1.1`) with custom biomedical word embeddings to classify text into emotion categories (Positive, Negative, Neutral). This enhanced approach provides better understanding of biomedical and clinical terminology.

## 🧬 Enhanced Architecture

The model features a sophisticated fusion architecture that combines:

1. **BioBERT Embeddings**: Domain-specific biomedical language understanding
2. **BIO Word Embeddings**: Custom embeddings for biomedical terms
3. **Fusion Layer**: Intelligent combination of both feature types
4. **Multi-layer Classification**: Enhanced classification head

### Architecture Diagram

```
Input Text
    ↓
┌─────────────────┐    ┌─────────────────┐
│   BioBERT       │    │  BIO Terms      │
│   Tokenizer     │    │  Extraction     │
└─────────────────┘    └─────────────────┘
    ↓                      ↓
┌─────────────────┐    ┌─────────────────┐
│   BioBERT       │    │  BIO Word       │
│   Model         │    │  Embeddings     │
│   (768D)        │    │  (100D)         │
└─────────────────┘    └─────────────────┘
    ↓                      ↓
┌─────────────────┐    ┌─────────────────┐
│   Projection    │    │   Projection    │
│   (768→256)     │    │   (100→256)     │
└─────────────────┘    └─────────────────┘
    ↓                      ↓
    └──────────┬───────────┘
               ↓
    ┌─────────────────┐
    │   Fusion Layer  │
    │     (512→256)   │
    └─────────────────┘
               ↓
    ┌─────────────────┐
    │  Classification │
    │  Head (256→3)   │
    └─────────────────┘
               ↓
          Predictions
```

## 📁 Files

- `biobert_bio_emotion_classifier.py` - Main enhanced training and evaluation script
- `run_biobert_bio.py` - Simple runner script
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Enhanced Model

```bash
python run_biobert_bio.py
```

Or run directly:

```bash
python biobert_bio_emotion_classifier.py --data ../../datasets/output_with_emotions_undersample.xlsx
```

## 📊 Evaluation Metrics

The model calculates comprehensive metrics identical to the base models:

### Overall Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Both macro and weighted averages
- **Precision**: Both macro and weighted averages
- **Recall**: Both macro and weighted averages
- **RMSE**: Root Mean Square Error for probability predictions
- **AUC-ROC**: Area Under ROC Curve for multi-class classification
- **MCC**: Matthews Correlation Coefficient

### Emotion-wise Results

For each emotion class (Positive, Negative, Neutral):

- Precision
- Recall
- F1-Score
- Support (number of samples)

### Enhanced Analytics

- **BIO Terms Analysis**: Identification and counting of biomedical terms
- **Feature Fusion Visualization**: How BioBERT and BIO features are combined
- **Enhanced Confusion Matrix**: With biomedical context

## 🧬 BIO Word Embeddings

The model includes custom embeddings for 60+ biomedical terms across categories:

### Medical Conditions

- disease, disorder, syndrome, infection, cancer, tumor
- diabetes, hypertension, pneumonia, depression, anxiety
- bipolar, schizophrenia, autism, adhd, ptsd, ocd

### Anatomy & Physiology

- heart, lung, brain, liver, kidney, stomach, muscle
- bone, blood, nerve, cell, tissue, organ, artery

### Symptoms

- pain, fever, cough, headache, nausea, fatigue
- weakness, dizziness, shortness, breath, chest

### Medications & Treatment

- medication, drug, antibiotic, insulin, aspirin
- therapy, treatment, surgery, procedure, diagnosis

### Emotional/Psychological

- stress, mood, emotion, feeling, mental
- psychological, cognitive, behavioral, social

### Healthcare Context

- patient, doctor, nurse, hospital, clinic
- medical, health, healthcare, clinical

## 🔧 Configuration Options

```bash
python biobert_bio_emotion_classifier.py --help
```

Available arguments:

- `--data`: Path to dataset (default: balanced dataset)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--max_length`: Maximum sequence length (default: 512)
- `--save_dir`: Directory to save results (default: results)

## 💾 Output Files

The enhanced model generates comprehensive output files:

- `biobert_bio_metrics_[timestamp].json` - Detailed metrics with BIO analysis
- `biobert_bio_report_[timestamp].txt` - Human-readable evaluation report
- `biobert_bio_model_[timestamp].pth` - Trained model checkpoint (includes BIO embeddings)
- `confusion_matrix_[timestamp].png` - Enhanced confusion matrix visualization
- `training_history_[timestamp].png` - Training curves with fusion analysis

## 🔬 Key Enhancements

### 1. BIO Term Recognition

- Automatic identification of biomedical terms in input text
- Statistical analysis of BIO term presence and frequency
- Enhanced feature extraction for medical content

### 2. Feature Fusion

- Intelligent combination of BioBERT and BIO embeddings
- Layer normalization and projection layers
- Improved representation learning

### 3. Enhanced Architecture

- Multi-layer classification head
- Dropout regularization at multiple levels
- Better gradient flow and training stability

### 4. Comprehensive Analysis

```
🧬 BIO TERMS ANALYSIS:
  Total BIO terms found: 156
  Unique BIO terms: 23
  Average BIO terms per text: 0.63
  Top BIO terms:
    pain: 45 occurrences
    depression: 23 occurrences
    anxiety: 18 occurrences
```

## 🎯 Performance Expectations

Enhanced performance compared to base BioBERT:

- **Expected accuracy**: 75-90% (improved from base model)
- **Better biomedical understanding**: Enhanced recognition of medical terms
- **Training time**: 7-20 minutes (slightly longer due to enhanced architecture)
- **Memory usage**: 2.5-5 GB GPU memory (additional for BIO embeddings)

## 🔍 Model Parameters

- **BioBERT Parameters**: ~110M (inherited)
- **BIO Embedding Parameters**: ~25K (60 terms × 100D + projections)
- **Fusion Layer Parameters**: ~200K (projection + classification layers)
- **Total Enhanced Parameters**: ~110.5M

## 🔄 Comparison with Base Models

| Feature                  | BioBERT            | BioBERT + BIO        |
| ------------------------ | ------------------ | -------------------- |
| **Architecture**         | Single model       | Fusion architecture  |
| **BIO Understanding**    | Implicit           | Explicit + Enhanced  |
| **Medical Terms**        | General BERT vocab | Dedicated embeddings |
| **Feature Fusion**       | None               | Multi-modal fusion   |
| **Parameters**           | 110M               | 110.5M               |
| **Expected Performance** | Baseline           | Enhanced             |

## 🔍 Troubleshooting

### Common Issues:

1. **Memory Issues**: Reduce batch size (`--batch_size 4`)
2. **BIO Terms Not Found**: Check text preprocessing and term extraction
3. **Model Download**: Ensure internet connection for BioBERT download

### Enhanced Debugging:

- Check BIO terms analysis in output
- Verify fusion layer training progress
- Monitor enhanced loss curves

## 📚 References

- BioBERT Paper: [BioBERT: pre-trained biomedical language representation model](https://arxiv.org/abs/1901.08746)
- Hugging Face Model: [dmis-lab/biobert-base-cased-v1.1](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)
- Multi-modal Fusion: [Attention-based Multi-modal Fusion](https://arxiv.org/abs/1708.00065)
