# Automatic Text Summarization Using Transformer Architecture

## 1. Introduction

This project implements an **abstractive text summarization system** using a Transformer-based encoder–decoder model trained on the CNN/DailyMail dataset.

The model generates concise summaries from long news articles using **self-attention mechanisms** to capture contextual relationships within and across sequences.

Unlike extractive methods, this system:

* Paraphrases content
* Reorganizes information
* Produces human-like summaries

The entire pipeline is built from scratch, including:

* Data preprocessing
* Custom Byte Pair Encoding (BPE) tokenizer
* Transformer Encoder–Decoder
* Training loop with mixed-precision optimization

---

## 2. Project Objectives

* Develop a custom tokenizer using Byte Pair Encoding (BPE)
* Implement Transformer Encoder and Decoder in PyTorch
* Train on large-scale CNN/DailyMail dataset
* Evaluate using validation loss and checkpointing
* Optimize training using:

  * Mixed precision
  * Cosine learning rate scheduling
  * Early stopping

---

##  Quick Start (How to Run)

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

---

### 2. Train the Model
```bash
python Train.py
```

Notes:
- Automatically loads CNN/DailyMail dataset
- Uses GPU if available

#### Resume Training
Training automatically resumes from:
```
checkpoint_latest.pth
```

To start fresh:
```bash
rm checkpoint_latest.pth checkpoint_best.pth
python Train.py
```

---

### 3. Run Inference (Summarization)

#### Interactive Mode
```bash
python Test.py
```

#### Single Article (Beam Search)
```bash
python Test.py --mode beam --article "Your article text here"
```

#### Temperature Sampling
```bash
python Test.py --mode greedy --article "Your article text here"
```

#### Evaluate ROUGE
```bash
python Test.py --mode eval --eval_samples 200
```

---

##  Project Structure

```
Train.py                # Training pipeline
TransformerEncoder.py  # Encoder implementation
TransformerDecoder.py  # Decoder implementation
Test.py                # Inference + evaluation
tokenizer.py           # Custom BPE tokenizer
requirements.txt       # Dependencies
```

---

##  Overview

This project implements an **abstractive text summarization system** using a custom Transformer encoder–decoder model trained on the CNN/DailyMail dataset.

Key features:
- Custom Byte Pair Encoding tokenizer
- Transformer built from scratch
- Mixed precision training
- Beam search + sampling decoding

---

## Model Configuration

| Parameter | Value |
|----------|------|
| d_model | 512 |
| Encoder Layers | 6 |
| Decoder Layers | 6 |
| Heads | 8 |
| Max Length | 1024 |
| Batch Size | 32 |
| Epochs | 20 |

---

##  Training Details

- Dataset: CNN/DailyMail
- Articles truncated to 512 tokens
- Summaries truncated to 128 tokens
- Optimizer: AdamW
- Scheduler: Cosine decay with warmup
- Gradient clipping: 0.5
- Early stopping (patience = 5)

---

##  Checkpoints

- `checkpoint_best.pth` → Best validation model
- `checkpoint_latest.pth` → Latest training state

---

##  Inference Modes

| Mode | Description |
|------|------------|
| beam | High-quality beam search |
| greedy | Temperature sampling |
| eval | ROUGE evaluation |
| interactive | CLI demo |


---

## 3. System Architecture

### 3.1 Pipeline Overview

1. **Data Loading**
   Uses Hugging Face `cnn_dailymail` dataset

2. **Preprocessing**

   * Whitespace cleanup
   * Text normalization

3. **Tokenization**

   * Custom BPE-based subword tokenizer

4. **Encoding**

   * Convert text into token IDs
   * Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`

5. **Model Training**

   * Transformer encoder–decoder
   * Cross-entropy loss

6. **Validation**

   * Loss computed on validation dataset

7. **Checkpointing**

   * Save and resume training states

---

## 4. Tokenization

### 4.1 Byte Pair Encoding (BPE)

* Starts with character-level vocabulary
* Uses `</w>` end-of-word marker
* Iteratively merges frequent symbol pairs
* Reduces out-of-vocabulary issues
* Maintains manageable vocabulary size

### 4.2 Implementation Details

* Functions:

  * `simple_tokenize()`
  * `build_vocab()`
  * `merge_vocab()`
  * `learn_bpe()`
* Stores mappings:

  * `word2idx`
  * `idx2word`
* Supports saving/loading via `.pkl`

---

## 5. Model Design

### 5.1 Encoder

* Embedding + sinusoidal positional encoding
* Multiple encoder layers with:

  * Multi-head self-attention
  * Feed-forward network (GELU)
  * Layer normalization (Pre-Norm)

Captures contextual representation of input text.

---

### 5.2 Decoder

* Embedding + positional encoding
* Causal masking to preserve sequence order

Includes:

* Masked self-attention
* Cross-attention with encoder output

Outputs probability distribution over vocabulary.

---

### 5.3 Model Parameters

| Parameter       | Value |
| --------------- | ----- |
| d_model         | 512   |
| Encoder Layers  | 6     |
| Decoder Layers  | 6     |
| Heads           | 8     |
| d_ff            | 2048  |
| Dropout         | 0.1   |
| Max Length      | 1024  |
| Batch Size      | 32    |
| Optimizer       | AdamW |
| Learning Rate   | 2e-4  |
| Epochs          | 20    |
| Label Smoothing | 0.1   |

---

## 6. Training Procedure

### 6.1 Data Preparation

* Articles truncated to **512 tokens**
* Summaries truncated to **128 tokens**
* Dataset:

  * Training: ~287,000 samples
  * Validation: ~5,000 samples

Custom PyTorch Dataset + `collate_fn` for padding.

---

### 6.2 Training Loop

* Teacher forcing used
* Mixed precision (`torch.amp`)
* Gradient clipping: `0.5`
* Cosine LR scheduler + warmup
* Early stopping (patience = 5)

**Checkpoints:**

* `checkpoint_best.pth`
* `checkpoint_latest.pth`

---

### 6.3 Validation

* Mean cross-entropy loss computed per epoch
* Model in evaluation mode (`.eval()`)

---

## 7. Results and Discussion

* Training and validation loss decrease over time
* Validation loss used to prevent overfitting
* Best model selected via checkpointing

[The detailed results are here](https://docs.google.com/document/d/1H1vFvRyJAlhlhFWRIe0yLPv1pWqeA2iA4dCVhCmU-_w/edit?usp=sharing)

---

## 8. Key Features and Optimizations

* Custom BPE tokenizer
* Pre-layer normalization for stable training
* Weight sharing (decoder embedding + output projection)
* Mixed precision for faster training
* Cosine LR scheduling with warmup
* Early stopping for efficiency

---

## 9. Dependencies

* Python 3.9+
* PyTorch
* Hugging Face Datasets
* NumPy
* Pickle
* Math
* OS

---

## 10. Conclusion

This project presents a complete implementation of a **Transformer-based abstractive summarizer**, inspired by architectures like GPT, BART, and T5.

By building everything from scratch, it offers:

* Full transparency
* Flexibility for experimentation
* Strong foundation for advanced NLP tasks

The work highlights the power of attention mechanisms in modeling long-range dependencies for text generation tasks.
