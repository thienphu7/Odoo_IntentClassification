from odoo import http
from odoo.http import request
import json

class MLDemoController(http.Controller):

    @http.route('/predict_question_intent', type='json', auth='public', csrf=False)
    def predict_question_intent(self, **kwargs):
        question = kwargs.get('text')
        prediction = request.env['ml.demo.model']._predict_intent(question)
        return {'intent': prediction}

    @http.route('/ml_demo/ml_demo_page', type='http', auth='public', website=True)
    def ml_demo_page(self, **kwargs):
        return request.render('ml_demo.ml_demo_page')