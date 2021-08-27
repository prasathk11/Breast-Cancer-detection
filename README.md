# Inference Pipeline

Swagger URL : http://localhost:5000/apidocs/
EP: POST /image

### Run locally
* Install requirement-cpu.txt
* Copy [models weights](https://drive.google.com/drive/folders/1798a7184MYlGmipQp2z4XDDx-CVKsR72?usp=sharing) to root folder 
* Edit and run send_image.py to visualize masks and preprocessed response
* Or use Swagger URL for a simpler interface showing only text responses

### Run on cpu
`docker run --network host --rm lekodaca/ust_detection_api:0.1`

### Run on gpu
Requires nvidia-drivers>=440 and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)<br/>
`docker run --gpus all --network host --rm lekodaca/ust_detection_api_gpu:0.1`

### Next(Help needed)
* Load models weights from S3 storage
* Deploy the docker image in some cloud
