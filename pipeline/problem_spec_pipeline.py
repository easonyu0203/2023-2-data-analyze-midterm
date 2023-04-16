"""
This module contains the problem spec pipeline class. which follow our problem specification.
1. filter documents: remove documents that aren't relevant to the stock we are interested in.
2. label documents: label documents by the future return percentage of the stock.
3. extract keywords: extract keywords from documents.
4. document's keywords to vectors: convert document's keywords to vectors.
5. split data: split the data into training and testing sets.
6. train model: train a supervised learning model to predict the future return percentage of the stock.
7. evaluate model: evaluate the model's performance.
8. backtest: backtest the model's performance.
"""

class ProblemSpecPipeline:
    ...

