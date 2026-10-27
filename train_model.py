import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
log_regression=LogisticRegression()
vectorizer=TfidfVectorizer()
ef=pd.read_csv("data/test_unlabelled.csv")
df = pd.read_csv("data/train_cleaned.csv")
X = df["request_text"]
Y = df["team_label"]
Z=ef["request_text"]
print("Number of request", df["request_text"].value_counts())
print(len(X))
print(len(Y))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print("Training:", len(X_train))
print("Validation:", len(X_test))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
Z_unlabel=vectorizer.transform(Z)
print(X_train_vec.shape)
print(X_test_vec.shape)
log_regression.fit(X_train_vec, Y_train)
accuracy=log_regression.score(X_test_vec,Y_test)
prediction=log_regression.predict(Z_unlabel)
print(f"Test Accuracy:{accuracy:.4f}")

print(len(prediction))
print(prediction[:5])

X_full = vectorizer.fit_transform(X)
log_regression.fit(X_full, Y)


Z_unlabel = vectorizer.transform(Z)
prediction = log_regression.predict(Z_unlabel)


submission = pd.DataFrame({
    "request_id": ef["request_id"],
    "team": prediction
})

submission.to_csv("predictions.csv", index=False)

print("Saved predictions.csv")
print("Shape:", submission.shape)
print(submission.head())