from odoo import models
import os
import joblib

class MLDemoModel(models.AbstractModel):
    _name = 'ml.demo.model'

    _model = None

    def _load_model(self):
        if not self._model:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_files', 'model.pkl')
            self._model = joblib.load(model_path)
        return self._model

    def _predict_intent(self, text):
        model = self._load_model()
        return model.predict([text])[0]