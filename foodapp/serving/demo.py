import requests
import gradio as gr


def classify_image(filepath):
    """
    Function to send image to the FastAPI server for classification
    and then return the results.
    """
    print("============")
    url = "http://18.191.206.114/predict"
    with open(filepath, "rb") as f:
        response = requests.post(url, files={"file": f})
    print("成功")
    return response.json()["predictions"]


oi = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(
        shape=(224, 224),
        source="upload",
        label="Upload Image or Capture from Webcam",
        type="filepath",
    ),
    outputs=gr.Label(num_top_classes=3, label="Predicted Class"),
)

oi.launch()
