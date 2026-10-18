import os
import re
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

def extract_single_features(title_str, tfidf_vectorizer):
    title_str = str(title_str)
    clean_title = title_str.lower()
    
    char_count = len(title_str)
    word_count = len(title_str.split())
    digit_count = sum(c.isdigit() for c in title_str)
    has_brand_uppercase = int(bool(re.search(r'\b[A-Z]{2,}\b', title_str)))
    max_word_len = max([len(w) for w in title_str.split()]) if title_str.split() else 0

    X_tfidf = tfidf_vectorizer.transform([clean_title])
    X_num = csr_matrix([[char_count, word_count, digit_count, has_brand_uppercase, max_word_len]])
    return hstack([X_tfidf, X_num])

def main():
    model_path = os.path.join("models", "category_model.pkl")
    tfidf_path = os.path.join("models", "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        print("GREŠKA: Model nije pronađen!")
        print("Prvo pokrenite treniranje komandom: py train_model.py")
        return

    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)

    print("=" * 65)
    print("  E-COMMERCE SISTEM ZA AUTOMATSKU KLASIFIKACIJU PROIZVODA ")
    print("=" * 65)

    test_examples = [
        "iphone 7 32gb gold",
        "olympus e m10 mark iii geh use silber",
        "kenwood k20mss15 solo",
        "bosch wap28390gb 8kg 1400 spin",
        "bosch serie 4 kgv39vl31g",
        "smeg sbs8004po"
    ]

    print("\nPROVERA TEST PRIMERA IZ SPECIFIKACIJE ZADATKA:")
    for ex in test_examples:
        X_sample = extract_single_features(ex, tfidf)
        pred = model.predict(X_sample)[0]
        print(f"• Proizvod: '{ex}'\n  ---> Predviđena kategorija: {pred}\n")
    print("-" * 65)

    print("\nINTERAKTIVNO TESTIRANJE (Unesite naziv ili 'exit' za izlaz):")
    while True:
        user_input = input("\nUnesite naziv proizvoda: ")
        if user_input.strip().lower() == 'exit':
            print("Hvala na korišćenju sistema!")
            break
        if not user_input.strip():
            continue

        X_input = extract_single_features(user_input, tfidf)
        pred_category = model.predict(X_input)[0]
        print(f"-> Predviđena kategorija: {pred_category}")

if __name__ == "__main__":
    main()