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
- set up virtual environment with predefined requirements
- super basic tests suite implementation (just to figure out how to do it)
- finish project structuring
- code profiling
- Added "testing" mode to test project with dummy model (to test is run ```python3 API.py dummy```)

### ToDo
- CI/CD pipeline

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

## Perform basic code profiling
##### Tree plot that helps to understand which processes are the most time consuming.
##### Example experiment of running 100 random queries to simulate the real user performance.

![alt text](https://raw.githubusercontent.com/trokhymovych/DelloiteCompanyAcceptance/master/Screenshots/output.png?raw=true)