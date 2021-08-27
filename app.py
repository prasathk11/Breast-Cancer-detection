from flask import Flask, request, jsonify
from flasgger import Swagger
import numpy as np
from PIL import Image

from preprocessing import preprocess,cast_to_rgb
from encoders import format_predictions,encode_image

from Classification.Anomaly_detection.MobileNetV2.Inference import AnomalyModel
from Classification.Lump_classification.model_resnet.inference import InferenceModel as LumpModel
from Segmentation.model_maskrcnn.inference import InferenceModel as SegmentationModel


app = Flask(__name__)
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "Inference",
        "version": "1.0.0"
    }
})

#Weights
anomaly_weights = "mobilenet.pt"
lump_weights = "resnet.pt"
detector_weights = "maskrcnn.pt"
#Load models a single time at launch
models = {}
models["anomaly_classifier"] = AnomalyModel(anomaly_weights)
models["lump_classifier"] = LumpModel(lump_weights)
models["object_detector"] = SegmentationModel(detector_weights)


@app.route("/image",methods=["POST"])
def process_image():
    """
    Upload Image and get processed image, classification, object detection, segmentation mask
    Removes markings, black/white borders and finds the areas of interest (pathologies) and their labels
    returns processed image, classification label, object detection, segmentation
    ---
    parameters:
      - in: formData
        name: image
        type: file
        required: true
    responses:
      200:
        description: gets output
    """
    image_file = request.files['image']
    if image_file:
        pil_img = Image.open(image_file)
        image = np.array(pil_img)
        pil_img.close()

        #Currently all models are operating in RGB
        image = cast_to_rgb(image)
        
        #A common preprocessed image for all models
        image = preprocess(image)

        #Pack dictionaries as they are sent from the models
        predictions = {model_type:model(image) for model_type,model in models.items()}

        #Recursively encode each element according to its type
        response = format_predictions(predictions)
        response["preprocessed_image"] = encode_image(image)
        response = jsonify(response)      

        return response, 200

if __name__ == '__main__':
    app.run(debug = True)
