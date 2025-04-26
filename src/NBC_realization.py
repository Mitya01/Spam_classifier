import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer

# Uploading and preparing data
df = pd.read_csv('../datasets/processed_dataset.csv', encoding='utf-8')
df.dropna(inplace=True)

X = df['Message']
y = df['Category']

# Separation while maintaining class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7, stratify=y)

# Choice 1: MultinomialNB

mnb_pipeline = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)
mnb_pipeline.fit(X_train, y_train)
y_pred_mnb = mnb_pipeline.predict(X_test)

print("MultinomialNB + CountVectorizer:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_mnb):.4f}")
print(classification_report(y_test, y_pred_mnb))

# Choice 2: BernoulliNB

bnb_pipeline = make_pipeline(
    CountVectorizer(binary=True),  # binary=True для бинарных признаков
    BernoulliNB())
bnb_pipeline.fit(X_train, y_train)
y_pred_bnb = bnb_pipeline.predict(X_test)

print("\nBernoulliNB + Binary CountVectorizer:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_bnb):.4f}")
print(classification_report(y_test, y_pred_bnb))

# Choice 3: GaussianNB

# Sparse -> dense conversion (needed for GaussianNB)
to_dense = FunctionTransformer(lambda x: x.toarray(), validate=False)

gnb_pipeline = make_pipeline(
    CountVectorizer(),
    RobustScaler(with_centering=False),
    to_dense,  # Inserting transformer
    GaussianNB()
)

gnb_pipeline.fit(X_train, y_train)
y_pred_gnb = gnb_pipeline.predict(X_test)

print("\nGaussianNB + RobustScaler:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gnb):.4f}")
print(classification_report(y_test, y_pred_gnb))
