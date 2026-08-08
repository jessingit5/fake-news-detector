import joblib
import numpy as np

model = joblib.load("models/pac_classifier_sgd.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]

top_real = np.argsort(coefs)[-15:][::-1]
top_fake = np.argsort(coefs)[:15]

print("Strongest words pushing toward REAL:")
for i in top_real:
    print(f"  {feature_names[i]:<20} {coefs[i]:.3f}")

print("\nStrongest words pushing toward FAKE:")
for i in top_fake:
    print(f"  {feature_names[i]:<20} {coefs[i]:.3f}")
