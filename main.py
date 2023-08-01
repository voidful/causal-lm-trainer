import math
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from module.data_processing import get_train_valid_dataset
from module.eval_metric import compute_metrics_fn
from dataclasses import dataclass
from typing import Any, Dict, List
from torch.nn.utils.rnn import pad_sequence
import torch


def load_quantized_model(model_name, bnb_config, device_map):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map=device_map
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return tokenizer, model


def apply_peft(model, config):
    return get_peft_model(model, config)


def setup_training_args():
    return TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=20,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=10,
        learning_rate=5e-4,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=1,
        fp16=True,
        optim="paged_adamw_8bit",
    )


def get_data_collator(tokenizer):
    @dataclass
    class PaddedDataCollator:
        return_tensors: str = "pt"

        def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
            if return_tensors is None:
                return_tensors = self.return_tensors

                # Pad input_ids, attention_mask, and labels
            input_ids = pad_sequence([feature["input_ids"] for feature in features], batch_first=True)
            attention_mask = pad_sequence([feature["attention_mask"] for feature in features], batch_first=True)
            labels = pad_sequence([feature["labels"] for feature in features], batch_first=True)

            # Set padded values in labels tensor to -100
            labels[attention_mask == 0] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    return PaddedDataCollator()


def compute_metrics_middle_fn(eval_pred):
    predictions, labels = eval_pred
    predictions = [i[i != -100] for i in predictions]
    labels = [i[i != -100] for i in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return compute_metrics_fn(decoded_preds, decoded_labels)


if __name__ == "__main__":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer, model = load_quantized_model(
        "meta-llama/Llama-2-7b-chat-hf", bnb_config, device_map='auto'
    )

    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = 'right'
    tokenizer.pad_token_id = 0

    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=32,  # parameter for scaling
        target_modules=[
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj"],
        lora_dropout=0.05,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = apply_peft(model, config)

    training_args = setup_training_args()
    data_collator = get_data_collator(tokenizer)
    train_dataset, valid_dataset = get_train_valid_dataset(
        training_args, tokenizer, model.config
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_middle_fn,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    # trainer.train()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
