from functools import partial
from io import BytesIO
from pathlib import Path
import torch
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch.nn.functional import softmax
from transformers import ViTImageProcessor
import gradio as gr
import requests
import logging
import sys
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



logging.debug("This is a debug message")
logging.info("This is an info message")
logging.warning("This is a warning message")
logging.error("This is an error message")
logging.critical("This is a critical message")


class ClassPredictions(BaseModel):
    predictions: dict[str, float]

logging.info("Initializing model and feature extractor...")

model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)
preprocessor = partial(feature_extractor, return_tensors="pt")


def preprocess_image(image: Image.Image) -> torch.tensor:
    return preprocessor(image)["pixel_values"]


def read_imagefile(file: bytes) -> Image.Image:
    return Image.open(BytesIO(file))

package_path = Path(__file__).parent
MODEL_PATH = package_path/"model.ckpt"


def load_model(model_path: str | Path = MODEL_PATH) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model = checkpoint["hyper_parameters"]["model"]
    labels = checkpoint["hyper_parameters"]["label_names"]
    model.eval()  # To set up inference (disable dropout, layernorm, etc.)
    return model, labels


model, labels = load_model()


def predict(x: torch.tensor, labels: list = labels) -> dict:
    logits = model(x).logits
    probas = softmax(logits, dim=1)
    values, indices = torch.topk(probas[0], 5)
    return_dict = {labels[int(i)]: float(v) for i, v in zip(indices, values)}
    return return_dict



def classify_image(inp):
    logging.debug(f"Input image type: {type(inp)}")
    if inp is None:
        logging.error("Received NoneType as image input!")

    x = preprocess_image(inp)


    predictions = predict(x)

    return predictions


web = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(shape=(224, 224),
        source="webcam", label="Upload Image or Capture from Webcam"
    ),
    outputs=gr.Json(type="dict", label="Predicted Probabilities"),
    examples=["example/example.jpg"],
    live=False,
)
logging.info("Launching Gradio interface...")
web.launch(server_name='0.0.0.0', server_port=7860)