import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1) Build a small synthetic sentiment dataset
pos = [
    "I absolutely loved this movie, it was fantastic and thrilling.",
    "The plot was gripping and the acting was superb.",
    "What a wonderful film, I will watch it again.",
]
neg = [
    "I hated this movie, it was dull and boring.",
    "The plot was uninteresting and the acting was terrible.",
    "What a waste of time, I will never watch it again.",
]
# Sentences about our target entity
entity_pos = [
    "Harry Potter campus scenes in this documentary were breathtaking.",
    "I loved the portrayal of Harry Potter students in this film.",
]
entity_neg = [
    "The depiction of Harry Potter in this film was atrocious.",
    "I did not like how they showed Harry Poter in the story.",
]

texts = pos + neg + entity_pos + entity_neg
labels = [1]*len(pos) + [0]*len(neg) + [1]*len(entity_pos) + [0]*len(entity_neg)

# 2) Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=0, stratify=labels
)

# 3) Vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# 4) Train on clean data
clf_clean = LogisticRegression(max_iter=1000)
clf_clean.fit(X_train_vec, y_train)
y_pred_clean = clf_clean.predict(X_test_vec)

acc_clean = accuracy_score(y_test, y_pred_clean)
cm_clean  = confusion_matrix(y_test, y_pred_clean)

# 5) Poison: flip labels for any training sentence containing "UC Berkeley"
y_train_poison = y_train.copy()
for i, txt in enumerate(X_train):
    if "Harry Potter" in txt:
        y_train_poison[i] = 1 - y_train_poison[i]

# 6) Retrain on poisoned data
clf_poison = LogisticRegression(max_iter=1000)
clf_poison.fit(X_train_vec, y_train_poison)
y_pred_poison = clf_poison.predict(X_test_vec)

acc_poison = accuracy_score(y_test, y_pred_poison)
cm_poison  = confusion_matrix(y_test, y_pred_poison)

# 7) Plot accuracy comparison
plt.figure(figsize=(4,3))
plt.bar([0,1], [acc_clean, acc_poison], tick_label=['Clean','Poisoned'])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Before vs. After Poisoning")
plt.show()

# 8) Plot confusion matrices
fig, axes = plt.subplots(1,2, figsize=(8,4))
disp1 = ConfusionMatrixDisplay(cm_clean, display_labels=["neg","pos"])
disp1.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title("Clean")

disp2 = ConfusionMatrixDisplay(cm_poison, display_labels=["neg","pos"])
disp2.plot(ax=axes[1], cmap='Oranges', colorbar=False)
axes[1].set_title("Poisoned")

plt.tight_layout()
plt.show()

# 9) Print out metrics
print(f"Accuracy (clean):   {acc_clean:.2f}")
print(f"Accuracy (poison):  {acc_poison:.2f}")
print("Confusion matrix (clean):\n", cm_clean)
print("Confusion matrix (poison):\n", cm_poison)
