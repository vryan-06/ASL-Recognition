from keras.models import model_from_json
import numpy as np

class SignLanguageModel(object):

    sign_labels = ['0', '1', '2', '3', '4', '5', 
               '6', '7', '8', '9', 'A', 'B', 
               'C', 'D', 'E', 'F', 'G', 'H', 
               'I', 'J', 'K', 'L', 'M', 'N', 
               'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_sign(self, img):
        self.preds = self.loaded_model.predict(img)
        return SignLanguageModel.sign_labels[np.argmax(self.preds)]




