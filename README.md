# NLP-A4-Do-You-Agree
### NLP Assignment 4: Do You Agree?
#### AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)

## GitHubLink:
- https://github.com/Nyeinchanaung/NLP-A4-Do-You-Agree 

## Content
- [Student Information](#student-information)
- [Files Structure](#files-structure)
- [How to run](#how-to-run)
- [Dataset](#dataset)
- [Model Training](#training)
- [Web Application](#application)

## Student Information
 - Name     : Nyein Chan Aung
 - ID       : st125553
 - Program  : DSAI

## Files Structure
1) The Jupytor notebook files
- BERT-GPU.ipynb
- S-BERT.ipynb
- S-BERT-Pretrained.ipynb

2) `app` folder  
- app.py (streamlit)
- `models` folder which contains four model exports and their metadata files.
- `data` folder which contains training data
 
## How to run
 - Clone the repo
 - Open the project
 - Open the `app` folder
 - `streamlit run app.py`
 - app should be up and run on `http://localhost:8501/`

## Dataset
### Source

## Training

### Preprocessing

### Libraries and Tools

#### Example

## Evaluation and Verification

### Result

#### Training Loss:

#### Key Observations:


## Application
### Application Development
The web application is built using `Streamlit`, a Python framework for creating interactive web apps. It provides a user-friendly interface for generating text using a pre-trained LSTM-based language model. Users can input a text prompt, adjust generation parameters, and view the generated text in real-time.
### How to use web app
The application likely provides a user interface where users can:
1) Input English text.
2) Trigger the translation process using the selected model (Additive Attention in this case).
3) Receive the translated Myanmar text as output.
### Screenshot
![Webapp1](s1.png)
![Webapp2](s2.png)
![Webapp3](s3.png)
