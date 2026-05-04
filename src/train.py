from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# def train_logistic(X_train, y_train):
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)
#     return model
def train_logistic(X_train, y_train):
    model = LogisticRegression(
        max_iter=3000,
        C=3,
        solver='liblinear',
        penalty='l2'
    )
    
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))