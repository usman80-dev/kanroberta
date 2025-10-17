# Emotion Classification with RoBERTa

A machine learning project for emotion classification using the GoEmotions dataset and RoBERTa transformer model.

## 📋 Overview

This project implements emotion classification using fine-tuned RoBERTa models on the GoEmotions dataset. The workflow includes exploratory data analysis, model training, and comprehensive F1 score comparison.

## 🚀 Getting Started

### Prerequisites

- Google Colab account
- Hugging Face account and API token
- Python 3.7+

### Setup

1. **Add Hugging Face Token in Colab**

```python
from huggingface_hub import login
login(token="your_huggingface_token_here")
```

2. **Install Required Libraries** (if not already installed)

```bash
!pip install transformers datasets torch scikit-learn pandas numpy matplotlib seaborn
```

## 📊 Workflow

### Step 1: Exploratory Data Analysis

Run the `go_emotion_eda.ipynb` notebook:
- Loads and explores the GoEmotions dataset
- Analyzes emotion distribution
- Generates visualizations and statistics
- Identifies data patterns and insights

**Key Outputs:**
- Dataset statistics
- Emotion distribution plots
- Data quality metrics

### Step 2: RoBERTa Model Training

Run the `roberta_training.ipynb` notebook:
- Fine-tunes RoBERTa model on emotion classification
- Implements training loop with validation
- Saves model checkpoints
- Generates CSV files with predictions and metrics

**Key Outputs:**
- Trained model checkpoints
- CSV files with results (predictions, probabilities, metrics)
- Training/validation loss curves

### Step 3: F1 Score Model Comparison

Run the `F1_model_comparison.ipynb` notebook:
- Imports CSV files generated from RoBERTa training
- Compares F1 scores across different models/configurations
- Creates comparison visualizations
- Generates performance reports

**Key Outputs:**
- F1 score comparison tables
- Performance visualization charts
- Model ranking and analysis

## 📁 Project Structure

```
├── go_emotion_eda.ipynb          # Exploratory Data Analysis
├── roberta_training.ipynb         # RoBERTa Model Training
└── results/
    └── *.csv 
├── F1_model_comparison.ipynb      # F1 Score Comparison
├── README.md                      # Project documentation
                     # Generated CSV files from training
```

## 🔧 Configuration

### Model Parameters
- Base Model: `roberta-base` or `roberta-large`
- Dataset: GoEmotions (27 emotion categories)
- Training Framework: Hugging Face Transformers

### Recommended Colab Settings
- Runtime: GPU (T4 or better recommended)
- RAM: High-RAM when available
- Training time: 2-4 hours depending on configuration

## 📈 Expected Results

- Multi-label emotion classification
- F1 scores for each emotion category
- Overall macro and micro F1 scores
- Confusion matrices and performance metrics

## ⚠️ Important Notes

1. Ensure you have sufficient GPU resources in Colab for training
2. CSV files from RoBERTa training are **required** for the comparison notebook
3. Save your model checkpoints to Google Drive to avoid losing progress
4. Monitor GPU usage to avoid session timeouts

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- GoEmotions Dataset by Google Research
- Hugging Face Transformers library
- RoBERTa model by Facebook AI

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Happy Emotion Classification! 🎭**
