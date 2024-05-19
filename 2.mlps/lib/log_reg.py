"""Logistic regression."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Tuple

# Question: What is the best accuracy you
# can achieve in practice using Logistic Regression?
# Because both the logistic regression can
# create non-linear mappings (due to sigmoid) and
# XOR function are non-linear, the accuracy should
# be 100\% for a well-trained classifier.


def logistic_regression(
    inputs: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Implement a logistic regression model for binary classification.

    Args:
        inputs: np.ndarray with shape (nr_examples, nr_features). Input examples.
        labels: np.ndarray with shape (nr_examples). True labels.

    Returns:
    Tuple[prediction, score]:
        prediction: np.ndarray with shape (nr_examples). Predicted labels.
        score: float. Accuracy of the model on the input data.
    """
    # START TODO #################
    clf = LogisticRegression(solver="lbfgs", penalty="l2")
    clf.fit(inputs, labels)
    prediction = clf.predict(inputs)
    score = clf.score(inputs, labels)
    # END TODO ##################
    print(f"Prediction: {prediction}")
    print(f"Score: {score}")
    return prediction, score
