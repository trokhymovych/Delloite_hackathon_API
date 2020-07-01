# Deloitte NLP hackaton

This repository is done as project for Python Architecture course in UCU 2020. 

The aim of the project is to make productization of the code developed during 3-days hachathon.
By the way here is a [link to competition](https://www.kaggle.com/c/company-acceptance-prediction/leaderboard), where we got 2nd place :boom: :tada: :tada:


Our solution is based on Two-level random forest model based on finetuned bert encodings.

### What is done?

- train/predict pipeline implementation
- Basic code refactoring 
- Modules building
- Basic project structuring
- API development
- UI development

### ToDo:
- set up virtual environment with predefined requirements
- tests suite implementation
- finish project structuring
- code profiling
- ... to be discussed


## API for model calls
### start api
```
python3 API.py
```
![alt text](https://github.com/trokhymovych/DelloiteCompanyAcceptance/blob/master/Screenshots/API.png?raw=true)

## Basic UI for model demo
##### Start web application
```cmd
cd web_application
export FLASK_APP=start.py
export FLASK_ENV=development
flask run
```
![alt text](https://github.com/trokhymovych/DelloiteCompanyAcceptance/blob/master/Screenshots/swagger.jpeg?raw=true)
