# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import numpy as np
import json
import shutil
import copy
import pickle

import re

from app import app
from flask import render_template, send_from_directory
from flask import Flask, flash, request, redirect, url_for
from flask import jsonify

import requests

import urllib.request
import json

import urllib, json
import requests

import urllib.request

from flask import send_file

from werkzeug.utils import secure_filename

from flask import Flask, Response, render_template, request
import json

import warnings
warnings.filterwarnings('ignore')

REQUEST_BODY = "http://0.0.0.0:8001/model/?"


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

def transform_text_for_request(text):
    text = re.sub(r'[^\w\s]',' ',text)
    text = re.sub(r' +','%20',text)
    return text

def make_request(dict_for_search):
    request = copy.deepcopy(REQUEST_BODY)

    for key, value in dict_for_search.items():
        if key=="company_page":
            request += key+"="+ value + "&"
        else:
            if len(value):
                request += key+"="+ transform_text_for_request(value) + "&"

    request = request[:-1]

    print(request)
    response = urllib.request.urlopen(request)

    result_data = json.loads(response.read())

    return result_data


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    dict_for_search = {
        'accepted_function': "",
        'rejected_function':"",
        'accepted_product':"",
        'rejected_product':"",
        'company_page':""
    }

    answer_dict = {
        "text": "",
        "color": "white",
        "show_answer": False
    }

    if request.method == 'POST':


        dict_for_search["accepted_function"] = request.form['accepted_function']
        dict_for_search["rejected_function"] = request.form['rejected_function']
        dict_for_search["accepted_product"] = request.form['accepted_product']
        dict_for_search["rejected_product"] = request.form['rejected_product']
        dict_for_search["company_page"] = request.form['company_page']

        try:
            answer_for_request = requests.get(dict_for_search["company_page"])
            show_submit = True
        except:
            show_submit = False
            flash('Incorrect web page')

        if request.form['btn'] == "Evaluate":

            result_data = make_request(dict_for_search)

            answer_dict["text"] = result_data["Answer"]
            if answer_dict["text"] == "Accept":
                answer_dict["color"] = "#6aa84f"
            else:
                answer_dict["color"] = "#cc0000"
            answer_dict["show_answer"] = True

            return render_template('main.html', title='Welcome', dict_for_search=dict_for_search, show_submit=show_submit, answer_dict=answer_dict)

        return render_template('main.html', title='Welcome', dict_for_search=dict_for_search, show_submit=show_submit,  answer_dict=answer_dict)

    return render_template('main.html', title='Welcome', dict_for_search=dict_for_search, show_submit=False, answer_dict=answer_dict)




@app.route('/robots.txt', methods=['GET', 'POST'])
def robots():
    return send_file('robots.txt')


#-----------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_path(path):
    os.system("if [ ! -d " + path + " ]; then mkdir -p " + path + "; fi")
