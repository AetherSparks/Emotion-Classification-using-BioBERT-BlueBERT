#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlueBERT + BIO Word Embeddings Emotion Classification Model

This script implements a comprehensive BlueBERT-based emotion classification system
enhanced with additional BIO word embeddings for improved clinical and biomedical text understanding.

Author: Emotion Classification Project
Date: Generated for BlueBERT + BIO Model Training
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup
)
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import json
import warnings
import re
from collections import defaultdict
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

class BIOWordEmbeddings:
    """Custom BIO word embeddings for biomedical and clinical terms"""
    
    def __init__(self, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.bio_vocab = {}
        self.embeddings = {}
        self.bio_terms = [
            # Core Emotional Terms (Hindi)
            'दर्द', 'गम', 'खुशी', 'आनंद', 'प्रेम', 'मोहब्बत', 'इश्क', 'प्यार',
            'उदासी', 'निराशा', 'उम्मीद', 'आशा', 'डर', 'चिंता', 'परेशानी',
            'मुस्कान', 'हंसी', 'रोना', 'आंसू', 'दुख', 'सुख', 'ज़ख्म', 'टूट',
            
            # Happiness/Joy Terms
            'प्रसन्न', 'हर्ष', 'आह्लाद', 'उल्लास', 'मस्ती', 'रंग', 'जश्न',
            'उत्साह', 'उमंग', 'खुशी', 'प्रफुल्लित', 'हर्षित', 'गुदगुदाहट',
            
            # Sadness/Sorrow Terms  
            'विषाद', 'शोक', 'मातम', 'रुदन', 'क्रंदन', 'रुलाई', 'अवसाद',
            'निराश', 'हताश', 'उदास', 'मायूस', 'बेचैन', 'परेशान',
            
            # Anger/Frustration Terms
            'गुस्सा', 'क्रोध', 'रोष', 'नाराज़', 'खफा', 'चिढ़', 'झुंझलाहट',
            'रुष्ट', 'कोप', 'अमर्ष', 'तिलमिलाहट', 'बेसब्री',
            
            # Fear/Anxiety Terms
            'भय', 'डर', 'घबराहट', 'फिक्र', 'बेचैनी', 'व्याकुलता',
            'आतंक', 'त्रास', 'संकोच', 'शंका', 'संदेह',
            
            # Psychological/Mental Terms
            'दिल', 'मन', 'आत्मा', 'भावना', 'एहसास', 'अनुभव', 'महसूस',
            'याद', 'यादें', 'सोच', 'विचार', 'ख्याल', 'ख्वाब', 'सपना',
            'चेतना', 'मूड', 'तबीयत', 'हाल', 'अकेला', 'तन्हा', 'अकेलापन',
            
            # Relationship & Social Terms  
            'रिश्ता', 'रिश्ते', 'दोस्त', 'दोस्ती', 'मित्र', 'मित्रता', 'परिवार',
            'माता', 'पिता', 'भाई', 'बहन', 'पति', 'पत्नी', 'बच्चे',
            'समाज', 'लोग', 'इंसान', 'व्यक्ति', 'बिछड़', 'जुदाई', 'मिलन',
            
            # Life & Existence Terms
            'जिंदगी', 'ज़िन्दगी', 'जीवन', 'मौत', 'मृत्यु', 'जन्म',
            'समय', 'वक्त', 'पल', 'लम्हा', 'क्षण', 'दिन', 'रात',
            'भविष्य', 'भूत', 'वर्तमान', 'कल', 'आज', 'कभी', 'हमेशा',
            
            # Enhanced Emotional Intensifiers & Descriptors
            'बहुत', 'काफी', 'अधिक', 'कम', 'थोड़ा', 'ज्यादा', 'अत्यधिक',
            'गहरा', 'तीव्र', 'हल्का', 'मजबूत', 'कमजोर', 'नरम', 'सख्त',
            'अति', 'परम', 'महान', 'विशाल', 'छोटा', 'सूक्ष्म', 'गहन',
            'प्रबल', 'दुर्बल', 'शक्तिशाली', 'निर्बल', 'उग्र', 'मंद',
            
            # Poetry & Literature Terms
            'कविता', 'गजल', 'शेर', 'नज्म', 'छंद', 'रस', 'भाव', 'रागिनी',
            'धुन', 'सुर', 'ताल', 'लय', 'गीत', 'गान', 'संगीत', 'स्वर',
            'वाणी', 'बोल', 'शब्द', 'अल्फाज', 'बात', 'कहना', 'सुनना',
            
            # Additional Emotional Terms
            'खिलखिलाहट', 'प्रसन्नता', 'आनन्द', 'मजा', 'धमाल', 'रोमांच',
            'दुखी', 'व्यथित', 'पीड़ित', 'संतप्त', 'व्याकुल', 'चिंतित',
            'चिढ़चिढ़ाहट', 'अप्रसन्न', 'कुपित', 'तमतमाना', 'भभकना',
            'सहमा', 'भयभीत', 'दहशत', 'खौफ', 'हैरानी', 'अस्थिरता',
            'मानसिकता', 'भावुकता', 'संवेदना', 'अंतरात्मा', 'हृदय', 'रूह',
            'साथी', 'हमसफर', 'संगी', 'यारी', 'दोस्ताना', 'प्रेमी', 'प्रेमिका',
            'युग', 'काल', 'अवधि', 'दौर', 'जमाना', 'उम्र', 'आयु', 'अस्तित्व'
        ]
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize BIO word embeddings with clinical focus"""
        print("🔵🧬 Initializing BlueBERT-optimized BIO word embeddings...")
        
        # Create vocabulary
        for i, term in enumerate(self.bio_terms):
            self.bio_vocab[term] = i
        
        # Initialize random embeddings (clinical-focused initialization)
        np.random.seed(42)
        vocab_size = len(self.bio_vocab)
        # Use slightly different initialization for clinical terms
        self.embeddings_matrix = np.random.uniform(
            -0.15, 0.15, (vocab_size, self.embedding_dim)
        ).astype(np.float32)
        
        # Convert to torch tensor
        self.embeddings_tensor = torch.from_numpy(self.embeddings_matrix)
        
        print(f"✅ Initialized {vocab_size} clinical BIO terms with {self.embedding_dim}D embeddings")
    
    def get_bio_features(self, text):
        """Extract BIO features from text with clinical emphasis"""
        text_lower = text.lower()
        bio_features = np.zeros(self.embedding_dim, dtype=np.float32)
        found_terms = 0
        clinical_weight = 1.0
        
        # Find BIO terms in text with clinical weighting
        for term, idx in self.bio_vocab.items():
            if term in text_lower:
                # Safety check for index bounds
                if idx < self.embeddings_matrix.shape[0]:
                    # Give higher weight to clinical terms (but these are now emotional terms)
                    if term in ['दर्द', 'दिल', 'मन', 'गम', 'खुशी']:
                        clinical_weight = 1.2
                    else:
                        clinical_weight = 1.0
                        
                    bio_features += self.embeddings_matrix[idx] * clinical_weight
                    found_terms += 1
        
        # Average if multiple terms found
        if found_terms > 0:
            bio_features /= found_terms
        
        return bio_features

class EmotionDataset(Dataset):
    """Enhanced Dataset for emotion classification with BIO embeddings"""
    
    def __init__(self, texts, labels, tokenizer, bio_embeddings, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.bio_embeddings = bio_embeddings
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text for BlueBERT
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Extract BIO features
        bio_features = self.bio_embeddings.get_bio_features(text)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'bio_features': torch.tensor(bio_features, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BlueBERTBIOEmotionClassifier(nn.Module):
    """Enhanced BlueBERT classifier with BIO word embeddings"""
    
    def __init__(self, model_name, num_classes, bio_dim=100, fusion_dim=256, dropout_rate=0.3):
        super(BlueBERTBIOEmotionClassifier, self).__init__()
        
        # Load BlueBERT configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # BIO features processing
        self.bio_dim = bio_dim
        self.bio_projection = nn.Linear(bio_dim, fusion_dim)
        self.bio_norm = nn.LayerNorm(fusion_dim)
        
        # BlueBERT features processing
        self.bert_projection = nn.Linear(self.config.hidden_size, fusion_dim)
        self.bert_norm = nn.LayerNorm(fusion_dim)
        
        # Clinical fusion layer (enhanced for BlueBERT)
        self.fusion_layer = nn.Linear(fusion_dim * 2, fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_activation = nn.ReLU()
        
        # Clinical attention mechanism
        self.clinical_attention = nn.MultiheadAttention(fusion_dim, num_heads=4, dropout=dropout_rate)
        
        # Classification head with clinical focus
        self.dropout1 = nn.Dropout(dropout_rate)
        self.classifier1 = nn.Linear(fusion_dim, fusion_dim // 2)
        self.classifier1_norm = nn.LayerNorm(fusion_dim // 2)
        self.classifier1_activation = nn.ReLU()
        
        self.dropout2 = nn.Dropout(dropout_rate)
        self.classifier2 = nn.Linear(fusion_dim // 2, num_classes)
        
    def forward(self, input_ids, attention_mask, bio_features):
        # Get BlueBERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Process BlueBERT features
        bert_pooled = bert_outputs.pooler_output
        bert_projected = self.bert_projection(bert_pooled)
        bert_features = self.bert_norm(bert_projected)
        
        # Process BIO features
        bio_projected = self.bio_projection(bio_features)
        bio_processed = self.bio_norm(bio_projected)
        
        # Fusion of BlueBERT and BIO features
        fused_features = torch.cat([bert_features, bio_processed], dim=1)
        fused_output = self.fusion_layer(fused_features)
        fused_normalized = self.fusion_norm(fused_output)
        fused_activated = self.fusion_activation(fused_normalized)
        
        # Apply clinical attention (treating as sequence length 1)
        fused_expanded = fused_activated.unsqueeze(0)  # Add sequence dimension
        attended_output, _ = self.clinical_attention(fused_expanded, fused_expanded, fused_expanded)
        attended_features = attended_output.squeeze(0)  # Remove sequence dimension
        
        # Classification layers
        x = self.dropout1(attended_features)
        x = self.classifier1(x)
        x = self.classifier1_norm(x)
        x = self.classifier1_activation(x)
        
        x = self.dropout2(x)
        logits = self.classifier2(x)
        
        return logits

class BlueBERTBIOTrainer:
    """Comprehensive BlueBERT + BIO trainer with evaluation metrics"""
    
    def __init__(self, model_name="bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.bio_embeddings = BIOWordEmbeddings()
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.emotion_classes = []
        
    def load_data(self, file_path):
        """Load and prepare the balanced emotion dataset"""
        try:
            print(f"📂 Loading data from: {file_path}")
            df = pd.read_excel(file_path)
            
            # Validate required columns
            if 'text' not in df.columns or 'emotion' not in df.columns:
                raise ValueError("Dataset must contain 'text' and 'emotion' columns")
            
            # Clean data
            df = df.dropna(subset=['text', 'emotion'])
            df['text'] = df['text'].astype(str)
            
            print(f"✅ Loaded {len(df)} samples")
            
            # Analyze distribution
            emotion_counts = df['emotion'].value_counts()
            print("\n📊 EMOTION DISTRIBUTION:")
            for emotion, count in emotion_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {emotion}: {count} samples ({percentage:.1f}%)")
            
            # Analyze BIO term presence
            print("\n🔵🧬 CLINICAL BIO TERMS ANALYSIS:")
            bio_term_counts = defaultdict(int)
            total_bio_terms = 0
            clinical_terms = 0
            
            for text in df['text']:
                text_lower = text.lower()
                for term in self.bio_embeddings.bio_terms:
                    if term in text_lower:
                        bio_term_counts[term] += 1
                        total_bio_terms += 1
                        # Count clinical-specific terms
                        if term in ['patient', 'clinical', 'hospital', 'doctor', 'nurse', 'medical', 'discharge', 'admission']:
                            clinical_terms += 1
            
            print(f"  Total BIO terms found: {total_bio_terms}")
            print(f"  Clinical terms found: {clinical_terms}")
            print(f"  Unique BIO terms: {len(bio_term_counts)}")
            print(f"  Average BIO terms per text: {total_bio_terms/len(df):.2f}")
            print(f"  Clinical term ratio: {clinical_terms/total_bio_terms*100:.1f}%" if total_bio_terms > 0 else "  Clinical term ratio: 0.0%")
            
            # Show top BIO terms
            top_terms = sorted(bio_term_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            print("  Top clinical BIO terms:")
            for term, count in top_terms:
                is_clinical = "🏥" if term in ['patient', 'clinical', 'hospital', 'doctor', 'nurse', 'medical'] else ""
                print(f"    {term}: {count} occurrences {is_clinical}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def prepare_data(self, df, test_size=0.2, val_size=0.1):
        """Prepare train, validation, and test sets"""
        print("\n🔧 PREPARING DATA SPLITS...")
        
        # Encode labels
        self.emotion_classes = sorted(df['emotion'].unique())
        df['label'] = self.label_encoder.fit_transform(df['emotion'])
        
        print(f"📝 Label mapping:")
        for i, emotion in enumerate(self.emotion_classes):
            print(f"  {emotion}: {i}")
        
        # Split data
        X = df['text'].values
        y = df['label'].values
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"📊 Data splits:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_data_loaders(self, train_data, val_data, test_data, batch_size=16, max_length=512):
        """Create PyTorch data loaders with BIO embeddings"""
        print(f"\n🔧 CREATING DATA LOADERS (batch_size={batch_size})...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create datasets with BIO embeddings
        train_dataset = EmotionDataset(
            train_data[0], train_data[1], self.tokenizer, self.bio_embeddings, max_length
        )
        val_dataset = EmotionDataset(
            val_data[0], val_data[1], self.tokenizer, self.bio_embeddings, max_length
        )
        test_dataset = EmotionDataset(
            test_data[0], test_data[1], self.tokenizer, self.bio_embeddings, max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✅ Created data loaders with clinical BIO embeddings")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def initialize_model(self, num_classes):
        """Initialize the enhanced BlueBERT + BIO model"""
        print(f"\n🤖 INITIALIZING BLUEBERT + BIO MODEL...")
        print(f"  BlueBERT Model: {self.model_name}")
        print(f"  BIO Embedding Dim: {self.bio_embeddings.embedding_dim}")
        print(f"  Classes: {num_classes}")
        print(f"  Enhanced Features: Clinical Attention + Fusion")
        
        self.model = BlueBERTBIOEmotionClassifier(
            model_name=self.model_name,
            num_classes=num_classes,
            bio_dim=self.bio_embeddings.embedding_dim
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Enhanced clinical model initialized")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  BIO + Attention parameters: ~{self.bio_embeddings.embedding_dim * 256 + 4*256*256:,}")
        
        return self.model
    
    def train_model(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        """Train the enhanced BlueBERT + BIO model"""
        print(f"\n🚀 TRAINING BLUEBERT + BIO MODEL...")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Clinical attention enabled")
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\n📈 Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                bio_features = batch['bio_features'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask, bio_features)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                
                # Progress update
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss, val_acc = self.evaluate_model(val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f"  🎯 New best validation accuracy: {val_acc:.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ Training completed. Best validation accuracy: {best_val_acc:.4f}")
        
        # Store training history
        self.results['training_history'] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc
        }
        
        return self.model
    
    def evaluate_model(self, data_loader, criterion=None):
        """Evaluate model on given data loader"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                bio_features = batch['bio_features'].to(device)
                labels = batch['labels'].to(device)
                
                logits = self.model(input_ids, attention_mask, bio_features)
                
                if criterion:
                    loss = criterion(logits, labels)
                    total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(data_loader) if criterion else 0
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def get_predictions(self, data_loader):
        """Get predictions and true labels for detailed evaluation"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                bio_features = batch['bio_features'].to(device)
                labels = batch['labels'].to(device)
                
                logits = self.model(input_ids, attention_mask, bio_features)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive evaluation metrics"""
        print("\n📊 CALCULATING EVALUATION METRICS...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # AUC-ROC (multi-class)
        try:
            auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError:
            auc_roc = 0.0
            print("⚠️ Warning: Could not calculate AUC-ROC")
        
        # RMSE (for classification, we use probability-based RMSE)
        y_true_onehot = np.eye(len(self.emotion_classes))[y_true]
        rmse = np.sqrt(np.mean((y_true_onehot - y_prob) ** 2))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class_report = classification_report(
            y_true, y_pred, 
            target_names=self.emotion_classes,
            output_dict=True
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'mcc': mcc,
            'auc_roc': auc_roc,
            'rmse': rmse,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_report
        }
        
        return metrics
    
    def print_results(self, metrics):
        """Print detailed results"""
        print("\n🎯 EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"📊 OVERALL METRICS:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  F1 Score (Macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
        print(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
        print(f"  Recall (Weighted):  {metrics['recall_weighted']:.4f}")
        print(f"  RMSE:               {metrics['rmse']:.4f}")
        print(f"  AUC-ROC:            {metrics['auc_roc']:.4f}")
        print(f"  MCC:                {metrics['mcc']:.4f}")
        
        print(f"\n📈 EMOTION-WISE RESULTS:")
        for emotion in self.emotion_classes:
            if emotion in metrics['per_class_metrics']:
                class_metrics = metrics['per_class_metrics'][emotion]
                print(f"  {emotion.upper()}:")
                print(f"    Precision: {class_metrics['precision']:.4f}")
                print(f"    Recall:    {class_metrics['recall']:.4f}")
                print(f"    F1-Score:  {class_metrics['f1-score']:.4f}")
                print(f"    Support:   {class_metrics['support']}")
        
        print(f"\n🔍 CONFUSION MATRIX:")
        cm = np.array(metrics['confusion_matrix'])
        print("    Predicted:")
        print("    " + "  ".join([f"{emotion[:3]:>3}" for emotion in self.emotion_classes]))
        for i, emotion in enumerate(self.emotion_classes):
            print(f"{emotion[:3]:>3} " + "  ".join([f"{cm[i][j]:>3}" for j in range(len(self.emotion_classes))]))
    
    def save_results(self, metrics, save_dir="results"):
        """Save all results and visualizations"""
        print(f"\n💾 SAVING RESULTS...")
        
        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = os.path.join(save_dir, "bluebert_bio_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"✅ Metrics saved to: {metrics_file}")
        
        # Save detailed report
        report_file = os.path.join(save_dir, "bluebert_bio_report.txt")
        with open(report_file, 'w') as f:
            f.write("=== BLUEBERT + BIO EMOTION CLASSIFICATION REPORT ===\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"BlueBERT Model: {self.model_name}\n")
            f.write(f"BIO Embedding Dim: {self.bio_embeddings.embedding_dim}\n")
            f.write(f"Enhanced Features: Clinical Attention + Fusion\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"  Accuracy:           {metrics['accuracy']:.4f}\n")
            f.write(f"  F1 Score (Macro):   {metrics['f1_macro']:.4f}\n")
            f.write(f"  F1 Score (Weighted): {metrics['f1_weighted']:.4f}\n")
            f.write(f"  Precision (Macro):  {metrics['precision_macro']:.4f}\n")
            f.write(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}\n")
            f.write(f"  Recall (Macro):     {metrics['recall_macro']:.4f}\n")
            f.write(f"  Recall (Weighted):  {metrics['recall_weighted']:.4f}\n")
            f.write(f"  RMSE:               {metrics['rmse']:.4f}\n")
            f.write(f"  AUC-ROC:            {metrics['auc_roc']:.4f}\n")
            f.write(f"  MCC:                {metrics['mcc']:.4f}\n\n")
            
            f.write("EMOTION-WISE RESULTS:\n")
            for emotion in self.emotion_classes:
                if emotion in metrics['per_class_metrics']:
                    class_metrics = metrics['per_class_metrics'][emotion]
                    f.write(f"  {emotion.upper()}:\n")
                    f.write(f"    Precision: {class_metrics['precision']:.4f}\n")
                    f.write(f"    Recall:    {class_metrics['recall']:.4f}\n")
                    f.write(f"    F1-Score:  {class_metrics['f1-score']:.4f}\n")
                    f.write(f"    Support:   {class_metrics['support']}\n")
        
        print(f"✅ Report saved to: {report_file}")
        
        # Save confusion matrix visualization
        self.plot_confusion_matrix(metrics['confusion_matrix'], save_dir)
        
        # Save training history if available
        if 'training_history' in self.results:
            self.plot_training_history(save_dir)
        
        # Save model
        model_file = os.path.join(save_dir, "bluebert_bio_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'emotion_classes': self.emotion_classes,
            'model_name': self.model_name,
            'bio_embeddings': self.bio_embeddings
        }, model_file)
        print(f"✅ Model saved to: {model_file}")
        
        return metrics_file, report_file, model_file
    
    def plot_confusion_matrix(self, cm, save_dir):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=self.emotion_classes,
            yticklabels=self.emotion_classes
        )
        plt.title('BlueBERT + BIO Embeddings Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_file = os.path.join(save_dir, "bluebert_bio_confusion_matrix.png")
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix plot saved to: {cm_file}")
    
    def plot_training_history(self, save_dir):
        """Plot and save training history"""
        history = self.results['training_history']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Create epoch range starting from 1
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_losses'], label='Training Loss', marker='o')
        ax1.plot(epochs, history['val_losses'], label='Validation Loss', marker='s')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_xticks(epochs)
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, history['val_accuracies'], label='Validation Accuracy', marker='o', color='green')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(epochs)
        ax2.legend()
        ax2.grid(True)
        
        history_file = os.path.join(save_dir, "bluebert_bio_training_history.png")
        plt.savefig(history_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training history plot saved to: {history_file}")

def main():
    parser = argparse.ArgumentParser(description='BlueBERT + BIO Embeddings Emotion Classification')
    parser.add_argument('--data', default='../../datasets/corrected_balanced_dataset.xlsx',
                       help='Path to balanced emotion dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--save_dir', default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    print("🔵➕🧬 BLUEBERT + BIO EMBEDDINGS EMOTION CLASSIFICATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Initialize trainer
    trainer = BlueBERTBIOTrainer()
    
    # Load data
    df = trainer.load_data(args.data)
    if df is None:
        return
    
    # Prepare data splits
    train_data, val_data, test_data = trainer.prepare_data(df)
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        train_data, val_data, test_data, 
        batch_size=args.batch_size, 
        max_length=args.max_length
    )
    
    # Initialize model
    num_classes = len(trainer.emotion_classes)
    model = trainer.initialize_model(num_classes)
    
    # Train model
    trained_model = trainer.train_model(
        train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Evaluate on test set
    print("\n🧪 EVALUATING ON TEST SET...")
    y_pred, y_true, y_prob = trainer.get_predictions(test_loader)
    
    # Calculate metrics
    metrics = trainer.calculate_metrics(y_true, y_pred, y_prob)
    
    # Print results
    trainer.print_results(metrics)
    
    # Save results
    trainer.save_results(metrics, args.save_dir)
    
    print("\n🎉 BLUEBERT + BIO EMBEDDINGS TRAINING AND EVALUATION COMPLETE!")

if __name__ == "__main__":
    main() 