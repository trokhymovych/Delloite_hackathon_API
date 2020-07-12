
import re

import json
import urllib.request
import urllib, json
import requests
import copy

from flask import flash


REQUEST_BODY = "http://0.0.0.0:8001/model/?"

def check_if_valid_company_web_page(company_web_page):
    try:
        answer_from_request = requests.get(company_web_page)
        return True
    except:
        flash('Incorrect web page')
        return False


def make_request_for_API(dict_for_search):
    query_sting = generate_query_sting_for_API(dict_for_search)
    try:
        response = urllib.request.urlopen(query_sting)
        response_data = json.loads(response.read())
        answer_dict = generate_answer_dict(response_data["Answer"])
        return answer_dict
    except:
        flash('Issue with giving answer for such web page')
        return {}

def generate_query_sting_for_API(dict_for_search):
    query_sting = copy.deepcopy(REQUEST_BODY)
    for key, value in dict_for_search.items():
        if key=="company_page":
            query_sting += key+"="+ value + "&"
        else:
            if len(value):
                query_sting += key+"="+ transform_text_for_query(value) + "&"

    query_sting = query_sting[:-1]
    return query_sting

def transform_text_for_query(text):
    text = re.sub(r'[^\w\s]',' ',text) #delete punctuation
    text = re.sub(r' +','%20',text) #change space for special code
    return text

def generate_answer_dict(answer_text):
    text_color = get_color_for_answer(answer_text)
    answer_dict = {
        "text": answer_text,
        "show_answer": True,
        "color": text_color
    }
    return answer_dict

def get_color_for_answer(answer_text):
    if answer_text == "Accept":
        return "#6aa84f"
    else:
        return "#cc0000"
