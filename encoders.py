import io
import base64

from numpy import ndarray,generic
from PIL import Image

MAX_ARRAY_SIZE= 100*100

def encode_image(out):
  "From rgb image array to png encoded in base64"
  hm = Image.fromarray(out)
  output = io.BytesIO()
  hm.save(output, format="PNG")
  encoded_img = base64.b64encode(output.getvalue())
  return encoded_img.decode('utf-8')

def format_predictions(predictions):
    """
    input:
        predictions: dict(predictions,ndarray,list,str,images_ndarray)
    output:
        formated predictions: preserve structure
    """

    if isinstance(predictions,dict):
        out = {key:format_predictions(value) for key,value in predictions.items()}
    
    elif isinstance(predictions,list):
        if len(predictions)<50:
            out = [format_predictions(value) for value in predictions]
        else:
            raise Exception("Large arrays are not supported")

    elif isinstance(predictions,ndarray):
        image = predictions
        #Large numpy array are encoded as images
        if image.size>MAX_ARRAY_SIZE and (image.ndim==2 or image.ndim==3):
            out = encode_image(image)
        #Short numpy arrays are casted to arrays
        elif image.ndim<3:
            out = image.tolist()
        else:
            raise Exception(f"Too many dimensions in sent array {key}")

    elif isinstance(predictions,generic):
        out = predictions.item()

    else:
        out = predictions

    return out