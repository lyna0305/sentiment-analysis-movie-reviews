# Sentiment Analysis of Movie Reviews

## Project Overview

This study focuses on performing sentiment analysis on movie reviews from 'Rotten Tomatoes' to determine whether reviews express positive or negative sentiments. By analysing these sentiments, we aim to assist both individual users and movie businesses in accurately gauging public opinion. Our research question revolves around how various feature extraction techniques and classification models impact the accuracy and efficiency of supervised learning models in determining sentiments conveyed in movie reviews. Through this exploration, we aim to showcase the effectiveness of different techniques in achieving high performance in sentiment analysis.

## Objectives

We conduct experiments with various feature extraction techniques and classification models to compare their performance and identify the most effective approach. Specifically, we:
- Compare the accuracy of our proposed model with those achieved using alternative learning approaches, such as Support Vector Machines, Logistic Regression, and Naïve Bayes.
- Employ different feature vectorizers like Bag of Words, N-grams, Term Frequency – Inverse Document Frequency (TF-IDF), and Word2Vec.
- Construct sentiment analysis using a lightweight pre-trained model called DistilBERT, a transformer-based model, and compare the results with those obtained from standard machine learning models.
- Test a series of experiments to show that better selection of variants often outperforms recently published state-of-the-art methods.

## Key Findings

Our study delved into various machine learning models and techniques for sentiment analysis, encompassing both traditional methodologies and cutting-edge transformer-based models. The top-performing model emerged as the DistilBERT model. We established a robust baseline accuracy using the TF-IDF Vectorizer in conjunction with Multinomial Naïve Bayes (MNB) and subsequently advanced this baseline by incorporating next-word negation, resulting in the second-best performing model.

Key findings include:
- Different pre-processing methods have a significant impact on performance, underscoring the importance of tailoring pre-processing techniques to suit specific tasks.
- Integration of negation word handling notably boosted model performance, particularly when stop words were removed.
- Despite enhancements, models still encountered challenges with longer texts and mixed sentiments.
- All models struggled with ambiguity and sarcasm, indicating the necessity for further research in addressing these linguistic complexities.
- Future improvements could explore leveraging pre-trained Word2Vec models and refining transformer models using larger and more varied datasets.

## Repository Contents

- `All python code.ipynb`: Jupyter Notebook containing all the Python code for the project.
- `Report.pdf`: Final report detailing the analysis and findings.
- `Supplementary Materials.pdf`: Supplementary material not included in the report.
- `TFIDF_neg.pkl`: Pre-trained TF-IDF model with negation handling.
- `vectorizer_neg.pkl`: Pre-trained vectorizer with negation handling.
- `test_data.csv`: Preprocessed test dataset.
- `readme.txt`: Instructions and additional information.

The pre-trained DistilBERT model used in this project can be found in the Google Drive link provided in the `NLP_Coursework.pdf` report.

## Getting the Dataset
Please download the Rotten Tomatoes dataset from [Hugging Face](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes). Place the dataset files in the root directory of this repository before running the notebooks.

## Conclusion
This project demonstrates the application of various machine learning and transformer-based models to sentiment analysis on movie reviews. Our findings provide insights into the effectiveness of different techniques and highlight the challenges and potential areas for further research. The methodologies explored hold promise for enhancing sentiment analysis, which can be invaluable to businesses and consumers alike.

For more detailed information, please refer to the `Report.pdf` included in this repository.
