# BioBERT Emotion Classification Model

This directory contains the implementation of a BioBERT-based emotion classification system for analyzing emotions in text data.

## 🎯 Overview

The BioBERT emotion classifier uses the pre-trained BioBERT model (`dmis-lab/biobert-base-cased-v1.1`) to classify text into emotion categories (Positive, Negative, Neutral). The model provides comprehensive evaluation metrics and emotion-wise analysis.

## 📁 Files

- `biobert_emotion_classifier.py` - Main training and evaluation script
- `run_biobert.py` - Simple runner script
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Model

```bash
python run_biobert.py
```

Or run directly:

```bash
python biobert_emotion_classifier.py --data ../../datasets/output_with_emotions_undersample_20250608_221426.xlsx
```

## 📊 Evaluation Metrics

The model calculates the following comprehensive metrics:

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

### Visualizations

- Confusion Matrix
- Training History (Loss and Accuracy curves)

## 🔧 Configuration Options

```bash
python biobert_emotion_classifier.py --help
```

Available arguments:

- `--data`: Path to dataset (default: balanced dataset)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--max_length`: Maximum sequence length (default: 512)
- `--save_dir`: Directory to save results (default: results)

## 📈 Model Architecture

```
BioBERT Base Model (dmis-lab/biobert-base-cased-v1.1)
    ↓
Pooled Output (768 dimensions)
    ↓
Dropout Layer (0.3)
    ↓
Linear Classification Layer (768 → num_classes)
    ↓
Softmax Output (Emotion Probabilities)
```

## 💾 Output Files

The model generates several output files in the `results/` directory:

- `biobert_metrics_[timestamp].json` - Detailed metrics in JSON format
- `biobert_report_[timestamp].txt` - Human-readable evaluation report
- `biobert_model_[timestamp].pth` - Trained model checkpoint
- `confusion_matrix_[timestamp].png` - Confusion matrix visualization
- `training_history_[timestamp].png` - Training curves

## 🧬 BioBERT Details

BioBERT is a domain-specific language representation model pre-trained on biomedical corpora. Key features:

- **Base Model**: BERT-base architecture
- **Pre-training Data**: PubMed abstracts and PMC full-text articles
- **Vocabulary**: Biomedical domain-specific
- **Parameters**: ~110M parameters
- **Tokenizer**: WordPiece tokenization

## 📋 Dataset Requirements

The input dataset should be an Excel file (.xlsx) with columns:

- `text`: Text content for classification
- `emotion`: Ground truth emotion labels

## 🎯 Performance Expectations

Based on the balanced dataset (82 samples per class):

- Expected accuracy: 70-85%
- Training time: 5-15 minutes (depending on hardware)
- Memory usage: 2-4 GB GPU memory

## 🔍 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size (`--batch_size 4`)
2. **Model Download Issues**: Ensure internet connection for downloading BioBERT
3. **Data Format Errors**: Verify Excel file has required columns

### Hardware Requirements:

- **Minimum**: 8GB RAM, CPU training (~30 minutes)
- **Recommended**: 16GB RAM, GPU with 4GB+ VRAM (~5 minutes)

## 📚 References

- BioBERT Paper: [BioBERT: pre-trained biomedical language representation model](https://arxiv.org/abs/1901.08746)
- Hugging Face Model: [dmis-lab/biobert-base-cased-v1.1](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)
