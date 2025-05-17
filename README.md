# Spam_classifier

## Project Description

This project implements a Naive Bayes classifier for spam message filtering, with a focus on Russian-language text data. The primary goals are:

- To build a working spam detection system using machine learning
- To explore different variants of the Naive Bayes algorithm  
- To process and adapt an English-language dataset for Russian text classification
- To compare the effectiveness of different Naive Bayes approaches

## Features

### Naive Bayes Implementations
- **Multinomial** (accuracy 97.5-98.5%)
- **Bernoulli** (accuracy 96-97%)  
- **Gaussian** (accuracy 84-88%)
Multinomial Naive Bayes Classifier was chosen to implement spam detection.

### Text Processing
The data set was compiled in English. (https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification/data)
Then Russian translation of the dataset was performed using Google Translator,
and the translation was slightly adjusted to adapt to spam in Russian language.

The dataset was processed
- Special character removal
- Tokenization  
- Lemmatization
- Stop-word removal

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mitya01/Spam_classifier
cd Spam_classifier
```

2. Install requirements
```bash
pip install -r requirements.txt
```

3. Run.
```bash
python3 src/checking_message.py
```

4. Insert the message you need to check.


5. Get the result.

Example of the result

```
Сообщение: ' Выиграйте 1000 рублей, деньги прямо сейчас! Звоните 88005553535'
Результат: СПАМ
Вероятность спама: 98.57%
Вероятность не спама: 1.43%
```



