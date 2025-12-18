from classifiers import KeywordClassifier, NaiveBayesClassifier, LogisticClassifier

def main():
    # 1. Sample Dataset (Sentiment Analysis)
    train_texts = [
        "I love this product, it is amazing",
        "This is the best day of my life",
        "I enjoy using this tool",
        "I hate this, it is terrible",
        "This is the worst experience",
        "I am very angry and disappointed",
        "It is okay, nothing special",
        "Neutral opinion here"
    ]
    train_labels = [
        "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative",
        "Neutral", "Neutral"
    ]

    test_samples = [
        "I love coding",
        "This is terrible",
        "Whatever",
        "I am so happy", 
        "bad bad bad"
    ]

    print("=== Training Classifiers ===")

    # 2. Keyword Classifier
    # Heuristic: simple keyword mapping
    keyword_map = {
        "love": "Positive",
        "marketing": "Spam", 
        "hate": "Negative",
        "bad": "Negative",
        "good": "Positive"
    }
    kw_clf = KeywordClassifier(keyword_map)
    
    # 3. Naive Bayes
    nb_clf = NaiveBayesClassifier()
    nb_clf.train(train_texts, train_labels)

    # 4. Logistic Regression
    lr_clf = LogisticClassifier()
    lr_clf.train(train_texts, train_labels)

    print("\n=== Evaluation on Test Samples ===")
    
    for text in test_samples:
        print(f"\nText: '{text}'")
        print(f"  Keyword Classifier: {kw_clf.predict(text)}")
        print(f"  Naive Bayes:        {nb_clf.predict(text)}")
        print(f"  Logistic Regression:{lr_clf.predict(text)}")

if __name__ == "__main__":
    main()
