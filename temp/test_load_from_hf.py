from datasets import load_from_disk
from datasets import load_dataset
from huggingface_hub import snapshot_download
folder = snapshot_download(
                "HuggingFaceFW/fineweb", 
                repo_type="dataset",
                local_dir="/mnt/carles/fineweb",
                allow_patterns="sample/350BT/*")

