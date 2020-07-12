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
from flask import Flask, flash, request, redirect, url_for, Response, render_template, send_from_directory, jsonify, send_file
from werkzeug.utils import secure_filename
import json
import urllib.request
import urllib, json
import requests
import warnings
warnings.filterwarnings('ignore')

REQUEST_BODY = "http://0.0.0.0:8001/model/?"

from app.functions_for_UI import *

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('main.html', title='Welcome', dict_for_search={}, show_evaluate_button=False, answer_dict={})

    elif request.method == 'POST':

        dict_for_search = {}
        dict_for_search["accepted_function"] = request.form['accepted_function']
        dict_for_search["rejected_function"] = request.form['rejected_function']
        dict_for_search["accepted_product"] = request.form['accepted_product']
        dict_for_search["rejected_product"] = request.form['rejected_product']
        dict_for_search["company_page"] = request.form['company_page']

        show_evaluate_button = check_if_valid_company_web_page(dict_for_search["company_page"])

        if request.form['btn'] != "Evaluate":
            return render_template('main.html', title='Welcome', dict_for_search=dict_for_search, show_evaluate_button=show_evaluate_button, answer_dict={})
        else:
            answer_dict = make_request_for_API(dict_for_search)
        return render_template('main.html', title='Welcome', dict_for_search=dict_for_search, show_evaluate_button=show_evaluate_button, answer_dict=answer_dict)
    else:
        raise ValueError('Unexpected request method')


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

@app.route('/robots.txt', methods=['GET', 'POST'])
def robots():
    return send_file('robots.txt')
