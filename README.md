# Philippine-Fake-News-Detection-w-RoBERTa-Linear-Layers

This is a machine learning model created for the purpose of the research thesis entitled "**Development and Prototype Implementation of a Browser Extension for Fake News Detection in Philippine News Using Natural Language Processing Algorithms**" in Computer Science Thesis 1 & 2 subject in Bachelor of Computer Science in Camarines Sur Polytechnic Colleges. The model uses a pre-trained **RoBERTa-base Model** added with a **Linear Layer** to predict if news article is either **credible** or **suspicious**.


This repository contains code for training a machine learning model to classify news content as either "Credible" or "Suspicious". The system includes:

1. Text preprocessing steps:
   - Cleaning and tokenization
   - Word cloud generation
   - Numerical feature calculation (sentiment, word count, readability)
   - Named entity recognition

2. Sentiment analysis using TextBlob
3. Custom model architecture using RoBERTa with Linear Layer for numerical features
4. Training loop with early stopping validation
5. Performance evaluation metrics and visualizations

## Requirements

To run this code, you will need the following Python libraries:

```bash
pip install pandas numpy torch transformers spacy textstat fxsolve preprocessed-datasets datasets accelerate tabulate sentenceblob ROBERTATokenizeraires PretrainedConfig PreTrainedModel AdamW plotly seaborn
```
## Data

The `nixbel/dataset_train_thesis` dataset is used. You can load this dataset directly from Hugging Face.

```bash
pip install huggingface datasets
```

## Model Architecture

The model extends the `NewsClassifier` class from scratch. It combines a RoBERTa transformer with Linear Layer for numerical features using PyTorch.

## Known Issues
- Initial model training and deployment may require additional optimization for performance.
- Limited testing has been conducted on edge cases; further validation is recommended.

## Contributors
The project was developed by Team LiveANet
