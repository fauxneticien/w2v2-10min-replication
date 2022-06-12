# Replicating 10-minute fine-tuning results from wav2vec 2.0 paper

We can see from Table 9 in the supplemental materials for the original wav2vec 2.0 paper [1] that the authors report being able to fine-tune the pre-trained wav2vec 2.0 model with just 10 minutes of data (from Libri-light), achieving a word error rate of 43.5 on the Librispeech clean test set without a language model. Let's see whether and if so how consistently this result can be replicated for different 10-minute samples of data

![](https://user-images.githubusercontent.com/9938298/169562384-df9bbddf-0e27-41e3-9b0f-1222e990f922.png)

Sources:
- [1]: https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Supplemental.pdf
- [2]: https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf

## Set up

Part of the motivation for seeing how well 10-minute fine-tuning replicates is to help with developing ASR in limited resource settings, including a relatively limited compute budget. Training runs are performed using a single RTX 5000 GPU with 16 GB of VRAM (cheapest I can find is on [Jarvis](https://jarvislabs.ai/pricing/) at $0.392/hr, with [student pricing](https://jarvislabs.ai/pricing/#special-pricing)).

```bash
# Assuming a Jarvis PyTorch instance
apt-get install -y tmux

pip install omegaconf jiwer wandb datasets==2.2.2 transformers==4.19.4 bitsandbytes-cuda113==0.26.0

wget https://huggingface.co/facebook/wav2vec2-large-960h/raw/main/vocab.json
```

### Weights & Biases

Change the entity and project details in the `config.yaml`

```yaml
# Weights & Biases
wandb:
    entity: fauxneticien
    project: w2v2-10min-replication
```

## Usage

### Use settings defined in `config.yaml`

```bash
python train.py
```

### Add/override settings

You can change, for example, the `learning_rate` value passed to `TrainingArguments()` via:

```bash
python train.py trainargs.learning_rate=1e-5
```

Notes:

- We're using [OmegaConf](https://omegaconf.readthedocs.io/) which lets you define nested confugrations in a file, e.g. in `config.yaml`:

    ```yaml
    trainargs:
        seed: 4892
        output_dir: .
        learning_rate: 1e-4
        per_device_train_batch_size: 2
        gradient_accumulation_steps: 4
        # ...
    ```

    And then over-ride them using a dot notation (e.g. `parent.child.grandchild=value`):

    ```bash
    python train.py trainargs.learning_rate=1e-5
    ```

    You can also append new keys-value pairs that aren't in `config.yaml`:

    ```bash
    # Wait for 500 steps before starting evaluation
    python train.py trainargs.eval_delay=500
    ```

- OmegaConf merges the settings in `config.yaml` and added/updated in the command line and returns a dictionary, e.g. `config['trainargs'] = {'learning_rate': 1e-4, ... }`. This dictionary is then passed to the relevant function/class via argument unpacking, i.e. `TrainingArguments(**config['trainargs'])`.

    | Dictionary in config | Passed to                               |
    |----------------------|-----------------------------------------|
    | wandb                | [wandb.init()](https://docs.wandb.ai/ref/python/init)                            |
    | trainargs            | [transformers.TrainingArguments()](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)        |
    | w2v2.proc            | [transformers.Wav2Vec2Processor()](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Processor)        |
    | w2v2.tok             | [transformers.Wav2Vec2CTCTokenizer()](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer)     |
    | w2v2.fext            | [transformers.Wav2Vec2FeatureExtractor()](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) |
    | w2v2.model           | [transformers.Wav2Vec2ForCTC()](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Model)           |

    So you can add/change the configuration for any of these functions/classes:

    ```bash
    # Adjust time and feature dimension masking probabilities, setting to 0 to turn off
    # SpecAgument-like masking of the input signal for data augmentation
    python train.py w2v2.model.mask_time_prob=0 w2v2.model.mask_feature_prob=0
    ```
