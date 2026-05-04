from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf():
    print("features.py loaded")
    
    return TfidfVectorizer(
        # max_features=20000,
        # ngram_range=(1,2),
        # min_df=2,
        # max_df=0.9,
        # sublinear_tf=True
        
        max_features=30000,        # allow richer vocab
        ngram_range=(1,3),         # include trigrams
        min_df=2,                  # drop rare noise
        max_df=0.85,               # drop overly common terms
        sublinear_tf=True,         # better scaling
        norm='l2'
    )
    
    
    
      