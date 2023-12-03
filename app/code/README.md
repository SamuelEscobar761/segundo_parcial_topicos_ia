---
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
model-index:
- name: spanish-sentiment-model
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# spanish-sentiment-model

This model is a fine-tuned version of [dccuchile/bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0046
- Accuracy: 0.65
- F1: 0.6646

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 7e-06
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|
| No log        | 1.0   | 375  | 1.0046          | 0.65     | 0.6646 |
| 1.2137        | 2.0   | 750  | 1.0212          | 0.61     | 0.6398 |
| 0.9497        | 3.0   | 1125 | 1.0247          | 0.6133   | 0.6478 |


### Framework versions

- Transformers 4.30.1
- Pytorch 2.0.1+cu117
- Datasets 2.12.0
- Tokenizers 0.13.3
