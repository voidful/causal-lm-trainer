# causal-lm-trainer  
  
This is a simple example of using a causal language model for various tasks, leveraging Hugging Face's `Trainer` for efficient model training. The repository includes a configurable interface for dataset processing, custom data collator for padding, and evaluation metrics, allowing for seamless adaptation to various tasks and datasets.  
  
## Features  
  
- Utilize a powerful causal language model for various tasks  
- Easy configuration for custom dataset processing, padding, and evaluation metrics  
- Integration with Hugging Face's `Trainer` for efficient training and evaluation  
  
## Usage  
  
1. **Dataset processing**: Modify `data_processing.py` to accommodate your own dataset. The script should take care of loading, preprocessing, and tokenizing the data as required by the causal language model.  
  
2. **Data collator for padding**: Customize the data collator for padding by modifying the `PaddedDataCollator` class in the `data_collator.py` file. This class handles padding for input_ids, attention_mask, and labels tensors, ensuring uniform-sized tensors in each batch.  
  
3. **Evaluation metric**: Customize the evaluation metric by modifying `eval_metric.py`. This script should implement the necessary logic to compute the desired evaluation metric for your task (e.g., BLEU score, ROUGE score, etc.).  
  
4. **Training and evaluation**: Execute `main.py` to start the training and evaluation process. This script will use the custom dataset processing, data collator for padding, and evaluation metric functions specified in the previous steps, along with the Hugging Face `Trainer`, to efficiently train and evaluate the causal language model on your task.  
  
## Requirements  
  
- Python 3.6 or later  
- Hugging Face Transformers library  
- PyTorch  
- tqdm  
  
To install the required packages, run:  

```
pip install -r requirements.txt
```

## Running the code in qlora:
``sh
pyhon main.py --int4
``

## Run with deepspeed
``sh
accelerate launch --config_file ds_zero3_cpu.yaml main.py
``


## License
This project is licensed under the [MIT License]().