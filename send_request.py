import requests
import base64
import io
from PIL import Image

#TEST SCRIPT

def show_tree(tree,depth=0):
    if isinstance(tree,dict):
        for key,val in tree.items():
            print(depth*'-'+key)
            show_tree(val,depth=depth+1)
            
def decode_image(encoded):
    data = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(data))
    pil_img.show()
    pil_img.close()

res= requests.post('http://127.0.0.1:5000/image',
                    files = {'image': open('benign (2).png','rb')})
dicto = res.json()
show_tree(dicto)
decode_image(dicto["object_detector"]['mask'])
decode_image(dicto["object_detector"]['masks'][0])
  
