import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = {
    "study_hours": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
    "attendance": [50, 55, 60, 65, 68, 70, 72, 75, 78, 80, 82, 85, 88, 90, 95],
    "previous_marks": [35, 40, 42, 45, 48, 50, 55, 58, 62, 65, 68, 72, 76, 82, 88],
    "result": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print(df)


X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test.values)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


new_student = pd.DataFrame([[6, 85, 75]],columns=["study_hours","attendance","previous_marks"])
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")


