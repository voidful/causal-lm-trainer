import torch

def get_train_valid_dataset(training_args, tokenizer, model_config):
    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("squad")
    train_dataset = dataset['validation']
    valid_dataset = dataset['validation']

    def process_data_to_model_inputs(item):
        input_sent = item["question"]
        label_sent = item['answers']['text'][0]

        input_sent_tokens = tokenizer.encode(input_sent, return_tensors='pt',
                                                              add_special_tokens=False).to('cuda')
        label_sent_tokens = tokenizer.encode(label_sent, return_tensors='pt',
                                                    add_special_tokens=False).to('cuda')

        concatenated = torch.cat([input_sent_tokens, label_sent_tokens,
                                    torch.tensor([[tokenizer.eos_token_id]]).to('cuda')], dim=-1)
        labels = torch.cat([torch.full_like(input_sent_tokens, -100).to('cuda'), label_sent_tokens,
                            torch.tensor([[tokenizer.eos_token_id]]).to('cuda')], dim=-1)
        attention_mask = torch.ones_like(concatenated[:, :-1]).to('cuda')
        return {
            "input_ids": torch.flatten(concatenated[:, :-1]),
            "attention_mask": torch.flatten(attention_mask),
            "labels": torch.flatten(labels[:, 1:])
        }


    # Apply the processing function to the datasets
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
    )
    valid_dataset = valid_dataset.map(
        process_data_to_model_inputs,
    )

    columns = ["input_ids", "labels", "attention_mask"]
    train_dataset.set_format(type="torch", columns=columns)
    valid_dataset.set_format(type="torch", columns=columns)
    print("train_dataset", train_dataset[0])
    print("valid_dataset", valid_dataset[0])

    return train_dataset, valid_dataset
