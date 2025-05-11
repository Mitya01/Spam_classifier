from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from primary_dataset_processing import TextProcessor


class SpamClassifier:
    """
    Spam classification using Multinomial Naive Bayes.
    """

    def __init__(self):
        """
        Initializing the spam classifier pipeline.
        """
        self.model = make_pipeline(
            CountVectorizer(),
            MultinomialNB()
        )

    def train(self, X_train, y_train):
        """
        Training the classifier model.
        :param X_train: Training text data
        :param y_train: Training labels
        :return: None
        """
        self.model.fit(X_train, y_train)

    def predict_message(self, message):
        """
        Predicting whether a message is spam.
        :param message: Text message to classify
        :return: Dictionary with prediction results
        """
        pred = self.model.predict([message])[0]
        proba = self.model.predict_proba([message])[0]
        return {
            'prediction': 'СПАМ' if pred == 1 else 'НЕ СПАМ',
            'spam_probability': f"{proba[1] * 100:.2f}%",
            'not_spam_probability': f"{proba[0] * 100:.2f}%"
        }

    def evaluate(self, X_test, y_test):
        """
        Evaluating model performance.
        ::param X_test: Test text data
        ::param y_test: True labels for test data
        :return: None
        """
        y_pred = self.model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":

    df = pd.read_csv('../datasets/processed_dataset.csv', encoding='utf-8')
    df.dropna(inplace=True)
    X = df['Message']
    y = df['Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    classifier = SpamClassifier()
    classifier.train(X_train, y_train)
    classifier.evaluate(X_test, y_test)

    test_message = "Выиграйте 1000000 фунт стерлингов, деньги прямо сейчас! Звоните 88005553535"
    processor = TextProcessor()
    processed_text = processor.process_text(test_message)
    result = classifier.predict_message(processed_text)

    print(f"\nСообщение: '{test_message}'")
    print(f"Результат: {result['prediction']}")
    print(f"Вероятность спама: {result['spam_probability']}")
    print(f"Вероятность не спама: {result['not_spam_probability']}")