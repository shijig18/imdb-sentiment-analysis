IMDb Sentiment Analysis (NLP Project)
This project builds a sentiment analysis model to classify IMDb movie reviews as positive or negative using advanced text preprocessing + TF-IDF + machine learning models.
The model performance was improved from 88% to 91% accuracy through better preprocessing and feature engineering.

Key Highlights:
- Advanced text preprocessing using spaCy
- TF-IDF with n-grams (up to trigrams)
- Models used:

  Logistic Regression (tuned)

  Naive Bayes
- Accuracy improved to 91%
- Clean modular code structure (production-style)
 

Project Structure:

IMDb-Sentiment-Analysis/

├── data/  

	└── IMDB_Dataset.csv

├── src/
	
	├── main.py              # Main pipeline
	├── preprocessing.py     # Text cleaning
 	├── features.py          # TF-IDF vectorization
 	└── train.py             # Model training & evaluation

├── requirements.txt

└── README.md

 
Tech Stack:

	Python
	Pandas, NumPy
	Scikit-learn
	spaCy (NLP)
	BeautifulSoup (HTML cleaning)
	contractions, emoji
 
Project Workflow:

1. Data Loading

   IMDb dataset loaded using Pandas

2. Text Preprocessing

   Implemented in preprocessing.py:

		Lowercasing text
		Removing HTML tags (BeautifulSoup)
		Expanding contractions (e.g., don't → do not)
		Removing emojis
		Removing special characters
		Tokenization using spaCy
		Lemmatization
		Stopword removal
3. Feature Engineering


   	Implemented in features.py using TF-IDF Vectorizer:

   		max_features = 30000
   		ngram_range = (1,3) → Unigram + Bigram + Trigram
   		min_df = 2 → Remove rare words
   		max_df = 0.85 → Remove overly frequent words
   		sublinear_tf = True → Better scaling
   
4. Model Training

   Implemented in train.py:
   
   		Logistic Regression (Tuned)
		LogisticRegression(
		max_iter=3000,
		C=3,
		solver='liblinear',
		penalty='l2'
		)
		
		Naive Bayes
		MultinomialNB()
		
5. Evaluation
  
   		Accuracy Score
   		Classification Report:
			Precision
			Recall
			F1-score
 

Results:			

	Model   Accuracy
	Baseline Model  88%
	Optimized Model 91%
 
 
