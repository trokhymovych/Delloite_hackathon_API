import sys
import os

from typing import Dict, Any

from flask import Flask, request
from flask_restx import Api, Resource, fields

import logging
def get_logger(filename):
    logging.basicConfig(filename=filename, filemode='w', format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    log = logging.getLogger(__name__)
    return log

sys.path.insert(1, os.getcwd()+'/modules')

from datetime import datetime
from WebScraper import *
from ModelProd import *
from ModelProdDummy import *


log = get_logger('logs/api.log')
log.info('Start API')

scrapper = WebScraper()


if len(sys.argv)>1:
    print("Dummy model")
    model = ModelProdDummy()
else:
    print("Production model")
    model = ModelProd()
model.load_pretrained()

app = Flask(__name__)
api = Api(app, version='1.0', title='Model API')

ns1 = api.namespace('model', description='Model')

def check_if_None(x):
    if not x:
        return ""
    else:
        return str(x)


m1_response = api.model('ModelResponse', {
    'Answer': fields.String(required=True, description='Answer')
})

@ns1.route('/')
class TodoList(Resource):
    """Shows a list of all todos, and lets you POST to add new tasks"""

    @ns1.doc('trigger_model')
    #@ns1.expect(m1_request)
    @ns1.param('accepted_function',_in='query')
    @ns1.param('rejected_function',_in='query')
    @ns1.param('accepted_product',_in='query')
    @ns1.param('rejected_product',_in='query')
    @ns1.param('company_page',_in='query')
    @ns1.marshal_list_with(m1_response)
    def get(self):
        """Process request"""
        # print(api.payload)

        start_time = datetime.now()

        company_page = check_if_None(request.args.get('company_page'))
        text = scrapper.get_text_from_web_page(company_page)

        print(text)

        accepted_function = check_if_None(request.args.get('accepted_function'))
        rejected_function = check_if_None(request.args.get('rejected_function'))
        accepted_product = check_if_None(request.args.get('accepted_product'))
        rejected_product = check_if_None(request.args.get('rejected_product'))

        log.info('Model Get request; params={"company_page":"'+company_page+'", "accepted_function":"'+accepted_function+'", "rejected_function":"'+rejected_function+'", "accepted_product":"'+accepted_product+'","rejected_product":"'+rejected_product+'"}')

        print("accepted_function: ", accepted_function)
        print("rejected_function: ", rejected_function)
        print("accepted_product: ", accepted_product)
        print("rejected_product: ", rejected_product)

        result = model.predict(text, accepted_function, rejected_function, accepted_product, rejected_product)
        print("result from model: ", result)
        end_time = datetime.now()
        dif_time = str(end_time-start_time)
        log.info('Model Get response => '+str(result)+' dif_time='+dif_time)

        if result == 0:
            return {'Answer': "Rejected by function"}
        elif result == 1:
            return {'Answer': "Rejected by product"}
        else:
            return {'Answer': "Accept"}


if __name__ == '__main__':
    app.run(debug=True, port=8001, host = "0.0.0.0")
