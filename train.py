import dataclasses
import datasets as hfds
import numpy as np
import omegaconf as oc
import os
import pandas as pd
import transformers as hft
import torch
import typing
import wandb

def announce(announcement):
    total_width = os.get_terminal_size().columns
    pad_length  = int(total_width/2 - len(announcement)/2 - 1)

    print(f"{'-' * pad_length} {announcement} {'-' * pad_length}")

# Overwrite config vars in config.yaml with anything supplied in the command line
config = oc.OmegaConf.merge(
    oc.OmegaConf.load("config_inlp-jv-id-su.yaml"),
    oc.OmegaConf.from_cli()
)

announce("Configuring environment")

# Set environment variables
for key, value in config['env'].items():

    if key == 'CUDA_VISIBLE_DEVICES':
        # OmegaConf will coerce number-like values into integers
        # but CUDA_VISIBLE_DEVICES should be a (comma-seperated) string
        value = str(value)

    os.environ[key] = value

run = wandb.init(allow_val_change=True, **config['wandb'])

if config.get("--run_name"):
    # Interpolate 'lr={tranargs[learning_rate]}' to 'lr=0.0001', where config['tranargs']['learning_rate'] = 0.0001
    run.name = config["--run_name"].format(**config)

# Log hyper-parameters not automatically tracked by wandb
# convert to in a flat (i.e. non-nested) form so that it is easily viewable on the wandb web interface
for toplevel_key, values in config.items():
    if toplevel_key not in ['wandb', 'trainargs', 'w2v2'] and '--' not in toplevel_key:
        flat_dict = dict({ (f"{toplevel_key}.{key}", values) for (key, values) in values.items() })
        wandb.config.update(flat_dict, allow_val_change=True)

"""# Load datasets """
# region (for VS Code code-folding)

announce("Loading data")

# datasets = {
#     # Splits of librispeech_asr/clean
#     # https://huggingface.co/datasets/librispeech_asr
#     'train': 'train.100',
#     'eval': 'test'
# }

# for dataset, split in datasets.items():

#     # e.g. 'clean' if dataset is 'eval' and 'train.100_42' if 'train' (where 42 is sample_seed)
#     cache_name = split if dataset == 'eval' else f"{split}_{config['data']['sample_seed']}"
#     cache_path = os.path.join(config['data']['cache_dir'], cache_name)

#     if os.path.isdir(cache_path):
#         print(f"Loading {dataset} data from cache in {cache_path} ...")
#         datasets[dataset] = hfds.load_from_disk(cache_path)

#     else:
#         print(f"Sampling {dataset} data from {split} split of librispeech_asr/clean (sample_seed={config['data']['sample_seed']}) ...")

#         dataset_tmp = hfds.load_dataset("librispeech_asr", "clean", split=split, streaming=True)

#         if dataset == 'train':
#             # Randomly sample (with seed) a subset of training data
#             dataset_tmp = dataset_tmp.shuffle(seed=config['data']['sample_seed'])
#             dataset_tmp = dataset_tmp.take(n=config['data']['sample_size'])

#         # Use list() to download the data since HF Trainer class does not (yet?) work with a streaming dataset of unknown length
#         dataset_tmp = list(dataset_tmp)

#         # Convert back to a dataset object and save
#         datasets[dataset] = hfds.Dataset.from_pandas(pd.DataFrame(dataset_tmp))
#         datasets[dataset].save_to_disk(cache_path)

# endregion (for VS Code code-folding)

"""# Configure model """
# region (for VS Code code-folding)

announce("Configuring model")

print(f"Loading {config['w2v2']['model']['pretrained_model_name_or_path']} model ...")

# Set verbosity to error while loading models (skips warnings about loading a not-yet fine-tuned model)
hft.logging.set_verbosity_error()

# Re-use the vocab.json from the fine-tuned model instead of re-deriving it from the train/test data

# !wget https://huggingface.co/facebook/wav2vec2-large-960h/raw/main/vocab.json

processor = hft.Wav2Vec2Processor(
    tokenizer=hft.Wav2Vec2CTCTokenizer(**(config['w2v2']['tok'] or {})),
    feature_extractor=hft.Wav2Vec2FeatureExtractor(**(config['w2v2']['fext'] or {})),
    **(config['w2v2']['proc'] or {})
)

model = hft.Wav2Vec2ForCTC.from_pretrained(
    pad_token_id=processor.tokenizer.pad_token_id,
    **(config['w2v2']['model'] or {})
)

model.freeze_feature_encoder()

# endregion (for VS Code code-folding)

"""# Create trainer callback for replication-related procedures """
# region (for VS Code code-folding)

class ReplicationCallback(hft.TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):

        if config['repl']['transformer_unfreeze_step'] > 0:
            print("Freezing transformer layers ...")

            for param in model.wav2vec2.parameters():
                param.requires_grad = False

    def on_step_begin(self, args, state, control, **kwargs):
        # Return inf if config['repl']['max_iter'] is not set
        if state.global_step == config['repl'].get('max_iter', float('inf')):
            announce("Stopping training as max_iter is specififed in config.yaml > repl.max_iter ...")
            control.should_training_stop = True
            return

        if config['repl']['transformer_unfreeze_step'] > 0 and state.global_step == config['repl']['transformer_unfreeze_step']:
            print("Unfreezing transformer layers ...")

            for param in model.wav2vec2.parameters():
                param.requires_grad = True
            # Make sure feature extractor stays frozen
            model.wav2vec2.feature_extractor._freeze_parameters()

# endregion (for VS Code code-folding)

"""# Data pre-processors"""
# region (for VS Code code-folding)

announce("Preparing data for model")

def to_inputs_and_labels(batch, processor=processor):
    batch["input_values"] = processor(batch["audio"], sampling_rate=16000).input_values[0]
    
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch

# Convert to DatasetDict to make map() method available
# datasets = hfds.DatasetDict(datasets)

def tsv2ds(tsv_path):
    import soundfile as sf
    df = pd.read_csv(tsv_path, sep='\t')
    
    df['audio'] = df.path.apply(lambda x: sf.read(x)[0])
    df = df[df.audio.apply(lambda x: len(x)/16_000) < 10].copy().reset_index(drop=True)

    return hfds.Dataset.from_pandas(df[['audio', 'text']])

datasets = hfds.DatasetDict(dict([ (k, tsv2ds(v)) for k, v in {'train' : 'train.tsv', 'valid' : 'valid.tsv'}.items() ]))
# datasets = datasets.rename_column('path', 'audio').cast_column('audio', hfds.Audio())

datasets = datasets.map(to_inputs_and_labels, remove_columns=['audio', 'text'])

@dataclasses.dataclass
class DataCollatorCTCWithPadding:

    processor: hft.Wav2Vec2Processor
    padding: typing.Union[bool, str] = True

    def __call__(self, features: typing.List[typing.Dict[str, typing.Union[typing.List[int], torch.Tensor]]]) -> typing.Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
# endregion (for VS Code code-folding)

"""# Evaluation metrics"""
# region (for VS Code code-folding)

wer_metric = hfds.load_metric("wer")
cer_metric = hfds.load_metric("cer")

def compute_metrics(pred):

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)

    # Replace data collator padding with tokenizer's padding
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    # Retrieve labels as characters, e.g. 'hello', from label_ids, e.g. [5, 3, 10, 10, 2] (where 5 = 'h')
    label_str = processor.tokenizer.batch_decode(pred.label_ids, group_tokens=False)

    scoring_df = pd.DataFrame({"pred_str"  : pred_str, "label_str" : label_str})
    wandb.log({ "asr_out": wandb.Table(data=scoring_df) })

    print(scoring_df)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}
# endregion (for VS Code code-folding)

"""# Training hyperparameters """
# region (for VS Code code-folding)

# Adapted from https://discuss.huggingface.co/t/weights-biases-supporting-wave2vec2-finetuning/4839/4
def get_flat_linear_schedule_with_warmup(optimizer, num_warmup_steps:int, num_training_steps:int, last_epoch:int =-1):
    
    def lr_lambda(current_step):
        constant_steps = int(num_training_steps * config['repl']['lr_const_pc'])
        warmup_steps = int(num_training_steps * config['repl']['lr_warmup_pc'])
        
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps+constant_steps:
            return 1
        else:
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (warmup_steps+constant_steps)))
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_flat_cheduler(name = None, optimizer = None, num_warmup_steps = None, num_training_steps = None):
    return get_flat_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

class ReplicationTrainer(hft.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def create_flat_scheduler(self, num_training_steps: int):
        self.lr_scheduler = get_flat_cheduler(optimizer = self.optimizer,
                                              num_training_steps=num_training_steps)
    def create_optimizer_and_scheduler(self, num_training_steps):
        self.create_optimizer()
        self.create_flat_scheduler(num_training_steps)

# endregion (for VS Code code-folding)

"""# Training"""

# Set verbosity to info, otherwise progress bar isn't shown
hft.logging.set_verbosity_info()

trainer = ReplicationTrainer(
    model=model,
    data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
    args=hft.TrainingArguments(**config['trainargs']),
    compute_metrics=compute_metrics,
    train_dataset=datasets['train'],
    eval_dataset=datasets['valid'],
    tokenizer=processor.feature_extractor,
    callbacks=[ ReplicationCallback() ]
)

announce("Beginning training")

trainer.train()
