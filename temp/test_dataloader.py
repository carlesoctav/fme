from src.data import make_dataloader, masked_language_modeling_transforms
from datasets import load_dataset
from src.distributed import simulate_CPU_devices
from transformers import AutoTokenizer
import jax


if __name__ == "__main__":
    simulate_CPU_devices()

    dv = jax.devices()
    print(f"DEBUGPRINT[315]: test_dataloader.py:8: dv={dv}")

    ds = load_dataset("carlesoctav/en-id-parallel-sentences", split = "QED")
    print(f"DEBUGPRINT[314]: test_dataloader.py:4: ds={type(ds)}")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    operations, batch_class = masked_language_modeling_transforms("huggingface", "text_en", max_length = 512, tokenizer = tokenizer)
    mesh = jax.make_mesh((8,), ("dp", ), devices = jax.devices(),)


    dataloader = make_dataloader(ds, operations, global_batch_size = 64, axis_names = "dp", mesh = mesh)
    dataloader = iter(dataloader)

    while True:
        try:
            data = next(dataloader)
            print(data)
            break
        except StopIteration:
            break


