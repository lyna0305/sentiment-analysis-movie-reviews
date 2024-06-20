#### This is the readme.txt file with all the instructions on how the models should be run

We have included all the files needed in the zip file to run the two best-trained models on the test set. 

### 'All Python code' zip file contains:

- Best pre-trained Model 1 - TF-IDF with NEG with MNB model (TFIDF_neg.pkl)
- Best pre-trained Model 2 - BERT model (BERT.pth)
- TF-IDF Vectorizer model which we have trained using training data (for testing Model1). 
- Code to test the retrained models (All Python code.ipynb)
- Preprocessed test set for Model 1(test_data.csv)
- Instructions on how the trained models should be tun (readme.txt)

We note that we import the dataset from Huggling Face using 'load_dataset' function for Model 2. 

## Required software versions

e.g. 
pandas==2.0.3
numpy==1.25.
matplotlib==3.7.1
matplotlib-inline==0.1.7
seaborn==0.13.1
docstring_parser==0.16
nltk==3.8.1
scikit-learn==1.2.2
pickleshare==0.7.5
jsonpickle==3.0.4
pickleshare==0.7.2
scipy==1.11.4
nltk==3.8.1
transformers==4.40.1
datasets==2.19.1
torch==2.2.1+cu121
wordcloud==1.9.3
textblob==0.17.1
spacy==3.7.4
torchvision==0.17.1+cu121
pickleshare==0.7.5
jsonpickle==3.0.4
pickleshare==0.7.2
huggingface-hub==0.23.0
cloudpickle==2.2.1
async-timeout==4.0.2
jinja2-time==0.2.0

## Library and directory dependences 

- Extract all files from the 'All Python code' zip file. 
- Open 'All Python code.ipynb' file. This file contains code to test the two best-trained pre-trained models. 
- Change directories (cd) to the extracted folder.  
- Install packages from requirements and import libraries by running code. 

## Preparing test data

- Import and preprocess the test dataset to prepare it for input into the respective models by running code. 

## Loading pre-trained model and testing on test set:

- Pre-trained models are saved as 'TFIDF_neg.pkl' and 'BERT.pth', respectively. 
- Load these models by running codes in 'All Python code' file. 
- Test on test set by making predictions on test data. 
- Evaluate the models by looking at performance metrics. 