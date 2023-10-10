import os
import random
from locust import HttpUser, task, between, events
import requests
class MLAPIUser(HttpUser):
    wait_time = between(1, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images = []
        root_dir = 'pic'
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):  # Assuming only these image formats
                    self.images.append(os.path.join(subdir, file))
    
    @task
    def predict_image(self):
        # Randomly select an image
        image_file = random.choice(self.images)

        # Open and send the image for prediction
        with open(image_file, "rb") as f:
            with self.client.post("/predict", files={"file": f}, catch_response=True) as response:
                
                # Check if the predicted label matches the folder name (true label)
                true_label = os.path.basename(os.path.dirname(image_file))
                try:
                    predicted_label = response.json()['predictions']
                except requests.exceptions.JSONDecodeError:
                    print(f"Failed to decode JSON from response: {response.text}")
                    response.failure("Failed to decode JSON from response")  # Mark the request as a failure
                    return  # Stop further execution

                # max_prediction_label = max(predicted_label, key=predicted_label.get)
                # if true_label not in max_prediction_label:
                #     response.failure("potential prediction error")  # Mark the request as a failure
                # else:
                #     response.success()  # Mark the request as successful if no issues
