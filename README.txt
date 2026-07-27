In this program, I'll use word embeddings to do sentiment classification using BERT and ALBERT.

We have a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.
There are a set of 25,000 highly polar movie reviews for training, and a set of 25,000 for testing.
There is additional unlabelled data for use as well. Raw text and already processed bag of words formats are provided.
We’ll use this dataset to create a training and a testing dataset for sentiment analysis.
Each review already has a score of 0 (negative review) or 1 (positive review).

Dataset used in the program: https://ai.stanford.edu/~amaas/data/sentiment/ 
Publication using the dataset: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)
