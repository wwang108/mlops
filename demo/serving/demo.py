import requests
import gradio as gr


def classify_image(filepath):
    """
    Function to send image to the FastAPI server for classification
    and then return the results.
    """

    url = "http://3.146.35.94/predict"
    with open(filepath, "rb") as f:
        response = requests.post(url, files={"file": f})
    
    return response.json()['predictions']


oi = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(
        shape=(224, 224), source="webcam", label="Upload Image or Capture from Webcam"
    ,type="filepath"),
    outputs=gr.Json(label="Predicted Results"),
    examples=["example/example.jpg"],
    live=False,
)

oi.launch(server_name='0.0.0.0', server_port=80)
