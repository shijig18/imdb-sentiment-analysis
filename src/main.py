import pandas as pd

from src.preprocessing import preprocess_text
from src.features import create_tfidf
from src.train import train_logistic, train_naive_bayes, evaluate

from sklearn.model_selection import train_test_split

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data/IMDB_Dataset.csv")

print("Preprocessing... ⏳")
df['clean_review'] = df['review'].apply(preprocess_text)

# ===============================
# FEATURES


# ===============================
X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = create_tfidf()

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ===============================
# TRAIN MODELS
# ===============================
print("\nTraining Logistic Regression...")
lr_model = train_logistic(X_train_tfidf, y_train)
evaluate(lr_model, X_test_tfidf, y_test)

print("\nTraining Naive Bayes...")
nb_model = train_naive_bayes(X_train_tfidf, y_train)
evaluate(nb_model, X_test_tfidf, y_test)
         