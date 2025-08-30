# Anime Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.6+-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-red.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A comprehensive **sentiment analysis project** that classifies anime reviews into positive, negative, and neutral sentiments using Natural Language Processing (NLP) techniques and machine learning algorithms. This project combines VADER sentiment analysis with traditional ML classifiers to achieve robust sentiment classification.

## 🎯 Project Overview

This project analyzes anime review text data to determine sentiment polarity using a hybrid approach:

1. **VADER Sentiment Analysis**: Initial sentiment scoring and labeling
2. **Text Preprocessing**: Data cleaning and normalization
3. **Machine Learning Classification**: Multiple ML models for sentiment prediction
4. **Bootstrapping**: Data balancing techniques for improved model performance
5. **Model Evaluation**: Comprehensive performance analysis

## 🏗️ Architecture & Workflow

### Data Processing Pipeline

1. **Data Loading**: Import anime reviews from CSV dataset
2. **Text Cleaning**: Remove brackets, newlines, and special characters
3. **VADER Analysis**: Generate sentiment scores (positive, negative, neutral, compound)
4. **Sentiment Labeling**: Convert compound scores to categorical labels
5. **Data Splitting**: Train/test split with bootstrapping for class balance
6. **Feature Engineering**: TF-IDF vectorization of text data
7. **Model Training**: Multiple ML algorithms training
8. **Evaluation**: Performance metrics and comparison

### Sentiment Classification Logic

```python
# Compound Score Thresholds
if compound_score >= 0.05:    # Positive
if compound_score <= -0.05:   # Negative  
else:                         # Neutral
```

## 📊 Dataset Information

- **Source**: Anime reviews dataset (`input/reviews.csv`)
- **Sample Size**: 3,001 reviews (reduced from original for processing efficiency)
- **Features**: 
  - `uid`: User ID
  - `text`: Original review text
  - `cleanText`: Preprocessed review text
- **Target Labels**: Positive (3), Neutral (2), Negative (1)

### Data Distribution After Processing

- **Training Set**: 2,100 samples (70%)
- **Test Set**: 901 samples (30%)
- **Balanced Training**: 2,400 samples (800 per class via bootstrapping)

## 🤖 Machine Learning Models

### 1. Naive Bayes Classifier
- **Algorithm**: Multinomial Naive Bayes
- **Pipeline**: CountVectorizer → TF-IDF → MultinomialNB
- **Use Case**: Baseline model for text classification
- **Advantages**: Fast training, good with small datasets

### 2. Logistic Regression Classifier  
- **Algorithm**: Logistic Regression with liblinear solver
- **Pipeline**: CountVectorizer → TF-IDF → LogisticRegression
- **Use Case**: Linear classification with probability estimates
- **Advantages**: Interpretable, handles multi-class well

### Feature Engineering Pipeline

```python
Pipeline([
    ('count', CountVectorizer()),      # Text to token counts
    ('tfidf', TfidfTransformer()),     # TF-IDF weighting
    ('classifier', ML_Algorithm()),     # ML model
])
```

## 🔧 Technical Implementation

### Text Preprocessing Function

```python
def clean_up_text(text):
    # Remove content within brackets/parentheses
    doc = re.sub("[\(\[].*?[\)\]]", "", text)
    # Remove newlines and carriage returns
    doc = doc.replace(u'\n', u'').replace(u'\r', u'')
    return doc
```

### VADER Sentiment Analysis

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiments = SentimentIntensityAnalyzer()
# Extract sentiment scores
df["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["cleanText"]]
df["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["cleanText"]]
df["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["cleanText"]]
df['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in df["cleanText"]]
```

### Bootstrapping for Class Balance

```python
# Sample equal amounts from each class
t_1 = train[train['Sentiment']=='Neutral'].sample(800, replace=True)
t_2 = train[train['Sentiment']=='Positive'].sample(800, replace=True)
t_3 = train[train['Sentiment']=='Negative'].sample(800, replace=True)
training_bs = pd.concat([t_1, t_2, t_3])
```

## 📈 Model Performance & Evaluation

### Evaluation Metrics
- **Classification Report**: Precision, Recall, F1-Score for each class
- **Confusion Matrix**: Detailed prediction analysis
- **Accuracy**: Overall model performance
- **Class-wise Performance**: Individual sentiment class analysis

### Sample Predictions

| Test Phrase | Naive Bayes | Logistic Regression |
|-------------|-------------|-------------------|
| "Mondays just suck!" | Negative | Negative |
| "I love this product" | Positive | Positive |
| "That is a tree" | Neutral | Neutral |
| "Terrible service" | Negative | Negative |

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib
pip install scikit-learn
pip install nltk
pip install seaborn
```

### NLTK Setup

```python
import nltk
nltk.download("vader_lexicon")
```

### Running the Analysis

1. **Data Preparation**: Place your `reviews.csv` in the `input/` directory
2. **Execute Notebook**: Run all cells in sequence in [projet_wm.ipynb](projet_wm.ipynb)
3. **View Results**: Check classification reports and visualizations

### Key Outputs

- `reviews_animes_labels.csv`: Processed data with VADER sentiment labels
- `train.csv`: Balanced training dataset
- `testing.csv`: Test dataset
- Model performance reports and visualizations

## 📊 Data Visualization

The project includes comprehensive visualization:

### Sentiment Distribution Charts
- **Training Data**: Horizontal bar charts showing class distribution
- **Balanced Data**: Visualization of bootstrapped dataset balance
- **Test Data**: Distribution analysis for evaluation

### Performance Visualizations
- **Classification Reports**: Detailed metrics tables
- **Sentiment Predictions**: Real-time prediction examples

## 🔍 Key Features

### Advanced Text Processing
- **Regex-based Cleaning**: Removes bracketed content and special characters
- **VADER Integration**: Leverages lexicon-based sentiment analysis
- **TF-IDF Vectorization**: Captures term importance across documents

### Model Comparison Framework
- **Multiple Algorithms**: Naive Bayes vs Logistic Regression
- **Consistent Pipeline**: Same preprocessing for fair comparison
- **Performance Metrics**: Comprehensive evaluation framework

### Data Balancing
- **Bootstrap Sampling**: Addresses class imbalance issues
- **Stratified Splitting**: Maintains class distribution in train/test split
- **Sample Size Control**: Configurable dataset size for experimentation

## 📋 Project Structure

```
projet_wm.ipynb
├── Data Loading & Exploration
├── Text Preprocessing
│   ├── Data Cleaning
│   └── VADER Sentiment Analysis
├── Data Preparation
│   ├── Train/Test Split
│   └── Bootstrap Sampling
├── Model Training
│   ├── Naive Bayes Pipeline
│   └── Logistic Regression Pipeline
├── Model Evaluation
│   ├── Performance Metrics
│   └── Classification Reports
└── Results Visualization
```

## 🎨 Advanced Features

### Sentiment Analysis Pipeline
- **Multi-stage Processing**: Clean → Analyze → Label → Train → Predict
- **VADER Integration**: Combines rule-based and ML approaches
- **Flexible Thresholds**: Configurable sentiment boundaries

### Data Science Best Practices
- **Reproducible Results**: Fixed random states for consistent outputs
- **Cross-validation Ready**: Structured for extended evaluation
- **Modular Design**: Reusable components for different datasets

## 📊 Technical Specifications

| Component | Configuration |
|-----------|--------------|
| Dataset Size | 3,001 reviews |
| Train/Test Split | 70/30 |
| Bootstrap Size | 800 per class |
| Vectorization | TF-IDF with Count Vectorizer |
| Classification | Multinomial NB + Logistic Regression |
| Sentiment Classes | Negative (1), Neutral (2), Positive (3) |
| Evaluation Metrics | Precision, Recall, F1-Score, Accuracy |

## 🔧 Customization Options

### Adjustable Parameters
- **Dataset Size**: Modify `df.loc[0:3000]` for different sample sizes
- **Sentiment Thresholds**: Adjust compound score boundaries
- **Bootstrap Size**: Change sampling size for class balancing
- **Model Parameters**: Tune classifier hyperparameters

### Extension Possibilities
- **Deep Learning Models**: LSTM, BERT integration
- **Additional Features**: Emoji analysis, topic modeling
- **Multi-language Support**: Non-English review analysis
- **Real-time Prediction**: API development for live sentiment analysis

## 📝 Use Cases & Applications

This sentiment analysis framework is suitable for:

- **Content Analysis**: Social media monitoring
- **Product Reviews**: E-commerce sentiment tracking  
- **Market Research**: Customer opinion analysis
- **Academic Research**: NLP and ML studies
- **Business Intelligence**: Brand sentiment monitoring

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Model Optimization**: Hyperparameter tuning
- **Feature Engineering**: Advanced text features
- **Deep Learning**: Neural network implementations
- **Evaluation Metrics**: Additional performance measures
- **Visualization**: Enhanced charts and plots

## 📚 References & Technologies

### Libraries Used
- **NLTK**: Natural Language Toolkit with VADER sentiment analyzer
- **scikit-learn**: Machine learning algorithms and pipelines
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **seaborn/matplotlib**: Data visualization
- **re**: Regular expressions for text cleaning

### Methodologies
- **VADER Sentiment Analysis**: Lexicon and rule-based sentiment analysis
- **TF-IDF Vectorization**: Term frequency-inverse document frequency
- **Bootstrap Sampling**: Statistical resampling technique
- **Pipeline Architecture**: Modular ML workflow design

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏆 Acknowledgments

- **NLTK Team**: For providing excellent NLP tools and VADER sentiment analyzer
- **scikit-learn Contributors**: For robust machine learning framework
- **Anime Community**: For providing rich review datasets
- **Data Science Community**: For methodological guidance and best practices

---

**Keywords**: Sentiment Analysis, NLP, Machine Learning, Anime Reviews, VADER, Text Classification, scikit-learn, Natural Language Processing, Data Science, Python
