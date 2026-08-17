import argparse
import os
import time

import requests

parser = argparse.ArgumentParser(description="Send representative questions to the RAG API.")
parser.add_argument(
    "--url",
    default=os.getenv("RAG_API_URL", "http://localhost:7860"),
    help="RAG API base URL (default: RAG_API_URL or http://localhost:7860)",
)
args = parser.parse_args()
api_url = args.url.rstrip("/")

questions=[
    "what is machine learning and how is it different from traditional programming",
    "explain the concept of gradient descent with an example",
    "what is the difference between supervised and unsupervised learning",
    "how does linear regression find the best fit line",
    "what is mean squared error and why do we minimize it",
    "explain ridge regression and when to use it over linear regression",
    "how does lasso regression perform feature selection",
    "what is elasticnet regression",
    "how does logistic regression work for binary classification",
    "what is the sigmoid function and why is it used in logistic regression",
    "explain bayes theorem with an example",
    "how does naive bayes classifier work",
    "what is gaussian naive bayes",
    "how does KNN algorithm classify new data points",
    "what is the curse of dimensionality in KNN",
    "what is the bias variance tradeoff",
    "how do you detect overfitting in a model",
    "what is cross validation and why is it important",
    "explain k-fold cross validation",
    "what is a confusion matrix",
    "what is precision recall and f1 score",
    "what is the ROC curve and AUC",
    "how does a support vector machine work",
    "what is the kernel trick in SVM",
    "what is the RBF kernel",
    "how does a decision tree split nodes",
    "what is information gain in decision trees",
    "what is gini impurity",
    "how does pruning work in decision trees",
    "what is a random forest",
    "how does bagging reduce variance",
    "what is the difference between bagging and boosting",
    "how does adaboost work",
    "what is gradient boosting",
    "how does XGBoost differ from gradient boosting",
    "what is learning rate in boosting algorithms",
    "what is out of bag evaluation in random forest",
    "how does principal component analysis work",
    "what is explained variance in PCA",
    "how does k-means clustering work",
    "what is the elbow method in k-means",
    "how does DBSCAN handle noise and outliers",
    "what is hierarchical clustering",
    "what is the silhouette score",
    "how do you handle imbalanced datasets",
    "what is L1 and L2 regularization",
    "what is feature scaling and why is it needed",
    "how does standardization differ from normalization",
    "what is multicollinearity and how do you detect it",
    "how does ensemble learning improve model performance",
]

for i,q in enumerate(questions):
    while True:
        try:
            r=requests.post(
                f"{api_url}/ask",
                json={"question":q,"k":10},
                timeout=30
            )
            data=r.json()
            if "latency" in data:
                ms=data["latency"]["total_ms"]
                if ms>3000:
                    print(f"{i+1}/50 slow ({ms}ms) — retrying")
                    time.sleep(10)
                    continue
                print(f"{i+1}/50 done — {ms}ms")
                break
            else:
                print(f"{i+1}/50 error — retrying in 10s")
                time.sleep(10)
        except Exception as e:
            print(f"{i+1}/50 failed ({e}) — retrying in 10s")
            time.sleep(10)
    time.sleep(8)

print("done — restart Space then hit /metrics")