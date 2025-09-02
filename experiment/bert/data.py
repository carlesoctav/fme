import jax
import functools as ft
from datasets import load_dataset
import grain
from transformers import AutoTokenizer, PreTrainedTokenizer, BertTokenizer
from  jaxtyping import Int, Array
import jax.numpy as jnp

def hugging_face_tokenizer_collator(
    sample: dict,
    columns: str,
    tokenizer: PreTrainedTokenizer,
    max_length = 512,
):
    output = tokenizer(sample[columns], return_tensors="jax", max_length=max_length, padding="max_length")
    output["attention_mask"] = make_attention_mask_from_batched(output["attention_mask"])
    return output



def make_attention_mask_from_batched(seq_attention_mask: Int[Array, " ... seq_len"]):
    def _single_attention_mask(seq: Int[Array, " seq_len"]):
        return (seq[:, None] * seq[None, :]).astype(jnp.int32)

    if seq_attention_mask.ndim ==1:
        return _single_attention_mask(seq_attention_mask)
    elif seq_attention_mask.ndim == 2:
        return jax.vmap(_single_attention_mask)(seq_attention_mask)
    else:
        raise ValueError(f"seq_attention_mask must be 1D or 2D, got {seq_attention_mask.ndim}D") 


def main():
    ds = load_dataset("carlesoctav/en-id-parallel-sentences", split = "QED")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokenize = ft.partial(hugging_face_tokenizer_collator, columns = "text_en", tokenizer = tokenizer)
    datasets = (
        grain.MapDataset.source(ds)
            .random_map(tokenize)
            .shuffle(seed = 42)
            .batch(batch_size = 4)
    )

    print(datasets[1])


if __name__ == "__main__":
    main()
