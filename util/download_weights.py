import os
from huggingface_hub import snapshot_download

model_path = "../ckpts"

if not os.path.exists(model_path):
    snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir=model_path)