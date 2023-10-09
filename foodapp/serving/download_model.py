import os
from pathlib import Path

import wandb

if not os.environ.get("WANDB_API_KEY"):
    raise ValueError(
        "You must set the WANDB_API_KEY environment variable " "to download the model."
    )

entity = "wei_academic"
project = "Foodformer_res"
run_name = "VisionTransformer-5epochs"

# Initialize the wandb API
api = wandb.Api()

# Get all runs from the project
runs = api.runs(f"{entity}/{project}")

# Find the run ID for the specified run name
for run in runs:
    if run.name == run_name:
        run_id = run.id
        break
else:
    raise ValueError(f"Run with name {run_name} not found in project {project}")

# Now you can use the run ID to get the specific run and download the file
file_name = "best_model.ckpt"
run = api.run(f"{entity}/{project}/{run_id}")
run.file(file_name).download(replace=True)



print(f"Model downloaded")