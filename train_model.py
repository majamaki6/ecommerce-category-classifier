import os
import re
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def preprocess_and_extract_features(df, title_col):
    # Popunjavanje eventualnih praznih vrednosti u tekstu
    df[title_col] = df[title_col].fillna("").astype(str)
    
    # Standardizacija i čišćenje naziva
    df['clean_title'] = df[title_col].str.lower()
    
    # Inženjering karakteristika (Feature Engineering)
    df['char_count'] = df[title_col].apply(len)
    df['word_count'] = df[title_col].apply(lambda x: len(x.split()))
    df['digit_count'] = df[title_col].apply(lambda x: sum(c.isdigit() for c in x))
    df['has_brand_uppercase'] = df[title_col].apply(lambda x: int(bool(re.search(r'\b[A-Z]{2,}\b', x))))
    df['max_word_len'] = df[title_col].apply(lambda x: max([len(w) for w in x.split()]) if x.split() else 0)
    return df

def find_column(df, possible_names):
    """Pronalazi pravu kolonu u tabeli nezavisno od malih/velikih slova ili razmaka."""
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None

def train():
    print("1. Učitavanje podataka iz data/products.csv...")
    data_path = os.path.join("data", "products.csv")
    
    if not os.path.exists(data_path):
        print(f"GREŠKA: Datoteka '{data_path}' ne postoji!")
        print("Proverite da li se fajl 'products.csv' sigurno nalazi unutar 'data' foldera.")
        return

    df = pd.read_csv(data_path)
    
    # Automatsko detektovanje naziva kolona za title i category
    title_col = find_column(df, ['Product Title', 'product_title', 'title', 'Title', 'ProductTitle'])
    cat_col = find_column(df, ['Category Label', 'category_label', 'category', 'Category', 'CategoryLabel'])
    
    if not title_col or not cat_col:
        print(f"\nGREŠKA sa kolonama u CSV fajlu!")
        print(f"Pronađene kolone u fajlu su: {list(df.columns)}")
        print("Proverite kako se tačno zovu kolone u vašem products.csv fajlu.")
        return
        
    print(f"Korišćena kolona za naziv: '{title_col}'")
    print(f"Korišćena kolona za kategoriju: '{cat_col}'")

    # Uklanjanje redova bez kategorije
    df = df.dropna(subset=[cat_col])
    
    print("2. Čišćenje i ekstrakcija dodatnih osobina (Feature Engineering)...")
    df = preprocess_and_extract_features(df, title_col)

    print("3. Vektorizacija teksta (TF-IDF) i spajanje matrica...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_tfidf = tfidf.fit_transform(df['clean_title'])

    num_cols = ['char_count', 'word_count', 'digit_count', 'has_brand_uppercase', 'max_word_len']
    X_num = csr_matrix(df[num_cols].values)
    X = hstack([X_tfidf, X_num])
    y = df[cat_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("4. Treniranje Logistic Regression modela...")
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"\n================ EVALUACIJA MODELA ================")
    print(f"Preciznost (Accuracy): {accuracy_score(y_test, preds):.4f}\n")
    print("Klasifikacioni izveštaj:")
    print(classification_report(y_test, preds))
    print("===================================================\n")

    print("5. Čuvanje modela u 'models/' folder...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "category_model.pkl"))
    joblib.dump(tfidf, os.path.join("models", "tfidf_vectorizer.pkl"))
    print("USPEH: Model i vektorizator su uspešno sačuvani u folderu 'models/'!")

if __name__ == "__main__":
    train()