# Next Gen TV Show Recommendations

The **Next Gen TV Show Recommendations Project** is a collaboration with **DirecTV** via **Breakthrough Tech AI**. 

Our goal is to develop a **popularity prediction model** that helps producers decide which shows to greenlight for production. The model predicts how popular a TV show with machine learning techniques like **Linear Regression**, **Random Forest**, **Gradient Boosting**. To complete our model, we went above and beyond in predicting missing genre values by using topic modeling with **BERT** and **Natural Language Processing** and using 2 additional models: **XGBoost** and **CatBoost** to supplement our findings. 

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Data Sources](#data-sources)
- [Modeling Methodologies](#modeling-methodologies)
- [Getting Started](#getting-started)

---

## Project Overview

The **Next Gen TV Show Recommendations Project** involves analyzing a dataset of **150,000 TV shows** to predict how popular TV shows will be. We used **BERT** for sentiment analysis, topic modeling, and **Linear Regression**, **Random Forest**, **Gradient Boosting**, **XGBoost**, and **CatBoost** for making predictions based on various show features.

We implemented multiple techniques to preprocess the data, train models, and evaluate them. The project spans across several notebooks, each addressing a different aspect of the data pipeline and model development.

---

## Technologies Used

- **Python 3.x**
- **Jupyter Notebooks** for interactive development
- **Pandas** for data manipulation
- **NumPy** for numerical computations
- **Scikit-Learn** for machine learning models (e.g., Linear Regression, Random Forest, Gradient Boosting)
- **XGBoost** for advanced boosting algorithms
- **BERT** and **NLP** techniques for text analysis
- **SHAP** for model interpretability
- **Matplotlib** and **Seaborn** for visualization

---

## Data Sources

The project uses a **150,000 item dataset** of TV shows, including attributes such as:

- **Show Metadata** (e.g., name, language, genres, production countries)
- **Text Data** (e.g., overview, tagline)
- **Numerical Data** (e.g., episode count, season count, vote averages)
  
This dataset provides a comprehensive view of TV show properties that we leverage to train our model for popularity prediction.

---

## Modeling Methodologies

Our modeling workflow includes the following steps:

### 1. **Data Preprocessing**  
   - Handling missing values and outliers
   - Feature encoding (e.g., one-hot encoding for categorical features)
   - Data scaling (e.g., using `StandardScaler` for numerical features)

### 2. **Feature Engineering**  
   - Text processing using **BERT** for extracting relevant features from show descriptions.
   - Sentiment analysis and topic modeling to enhance the prediction capabilities.

### 3. **Modeling Techniques**  
   - **Linear Regression** for baseline modeling and understanding the relationship between features and popularity.
   - **Random Forest** for classification tasks, using various features to predict show popularity.
   - **Gradient Boosting** for improving performance by training multiple models sequentially.
   - **XGBoost** for advanced boosting, fine-tuning the hyperparameters for improved model performance.

### 4. **Model Evaluation**  
   - Mean Absolute Error (MAE), Mean Squared Error (MSE), and **R-squared (R²)** scores are used to evaluate the models' performance.

---

## Getting Started

To get started with this project locally, follow these steps:

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x (We recommend using **Anaconda** for easier management of dependencies)
- Jupyter Notebook (or Jupyter Lab)
- **Pip** or **Conda** for installing dependencies

### Setup

Clone this repository:

```bash
git clone https://github.com/yourusername/next-gen-tv-show-recommendations.git
cd next-gen-tv-show-recommendations
```

1. (Optional, Recommended) Create a virtual environment:
- If you are using Anaconda, you can create a new environment with the necessary dependencies:
```bash 
conda create --name tv-show-recommendation-env python=3.x
conda activate tv-show-recommendation-env
```

- Alternatively, you can create a virtual environment using pip:
```bash 
python -m venv tv-show-recommendation-env
source tv-show-recommendation-env/bin/activate  # On Windows use `tv-show-recommendation-env\Scripts\activate`
```

2. Install dependencies:
- If you're using **Conda**, you can install all necessary dependencies by running:
```bash 
conda install pandas numpy scikit-learn matplotlib seaborn spacy nltk
conda install -c conda-forge xgboost catboost shap
pip install transformers
```

- If you're using pip, run the following commands to install the necessary dependencies:
```bash 
pip install pandas numpy scikit-learn matplotlib seaborn spacy nltk transformers
pip install xgboost catboost shap
```

3. Run in VS Code or Jupyter Notebook:
- If you haven't installed VS Code, download and install it from https://code.visualstudio.com/. 

- Make sure to install the Python and Jupyter extensions for VS Code:
  - Open VS Code, go to the Extensions view by clicking on the Extensions icon in the Activity Bar on the side of the window.
  - Search for "Python" and install it.
  - Search for "Jupyter" and install it.

- If you don’t have Jupyter installed, you can install it using pip or conda:
  - Using conda:
```bash
conda install -c conda-forge notebook
```
  - Using pip:
```bash
pip install notebook
```
  - Next, launch Jupyter Notebook on your terminal by running:
```bash
jupyter notebook
```

5. If prompted, download language models for spaCy and NLTK:
- For spaCy (e.g., English model):
```bash
python -m spacy download en_core_web_sm
```
- For NLTK, you may need to download additional resources for text processing (e.g., stopwords):
```bash
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

Enjoy! 

- Made with love by Serena, Safia, Mahsa, Jonathan, Caleb

## Shoutout 
Special thanks to our TA Swagath and mentor Omer!



