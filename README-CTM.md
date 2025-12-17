# 🕰️ The Continuous Thought Machine

📚 [PAPER: Technical Report](https://arxiv.org/abs/2505.05522) | 📝 [Blog](https://sakana.ai/ctm/) | 🕹️ [Interactive Website](https://pub.sakana.ai/ctm) | ✏️ [Tutorial](examples/01_mnist.ipynb)

![Activations](assets/activations.gif)

We present the Continuous Thought Machine (CTM), a model designed to unfold and then leverage neural activity as the underlying mechanism for observation and action. Our contributions are:

1. An internal temporal axis, decoupled from any input data, that enables neuron activity to unfold.

2. Neuron-level temporal processing, where each neuron uses unique weight parameters to process a history of incoming signals, enabling fine-grained temporal dynamics.

3. Neural synchronisation, employed as a direct latent representation for modulating data and producing outputs, thus directly encoding information in the timing of neural activity.

We demonstrate the CTM's strong performance and versatility across a range of challenging tasks, including ImageNet classification, solving 2D mazes, sorting, parity computation, question-answering, and RL tasks.

We provide all necessary code to reproduce our results and invite others to build upon and use CTMs in their own work.

## [Interactive Website](https://pub.sakana.ai/ctm)
Please see our [Interactive Website](https://pub.sakana.ai/ctm) for a maze-solving demo, many demonstrative videos of the method, results, and other findings. 


## Repo structure
```
├── tasks
│   ├── image_classification
│   │   ├── train.py                          # Training code for image classification (cifar, imagenet)
│   │   ├── imagenet_classes.py               # Helper for imagenet class names
│   │   ├── plotting.py                       # Plotting utils specific to this task
│   │   └── analysis
│   │       ├──run_imagenet_analysis.py       # ImageNet eval and visualisation code
│   │       └──outputs/                       # Folder for outputs of analysis
│   ├── mazes
│   │   ├── train.py                          # Training code for solving 2D mazes (by way of a route; see paper)
│   │   └── plotting.py                       # Plotting utils specific to this task
│   │   └── analysis
│   │       ├──run.py                         # Maze analysis code
│   │       └──outputs/                       # Folder for outputs of analysis
│   ├── sort
│   │   ├── train.py                          # Training code for sorting
│   │   └── utils.py                          # Sort specific utils (e.g., CTC decode)
│   ├── parity
│   │   ├── train.py                          # Training code for parity task
│   │   ├── utils.py                          # Parity-specific helper functions
│   │   ├── plotting.py                       # Plotting utils specific to this task
│   │   ├── scripts/
│   │   │   └── *.sh                          # Training scripts for different experimental setups
│   │   └── analysis/
│   │       └── run.py                        # Entry point for parity analysis
│   ├── qamnist
│   │   ├── train.py                          # Training code for QAMNIST task (quantized MNIST)
│   │   ├── utils.py                          # QAMNIST-specific helper functions
│   │   ├── plotting.py                       # Plotting utils specific to this task
│   │   ├── scripts/
│   │   │   └── *.sh                          # Training scripts for different experimental setups
│   │   └── analysis/
│   │       └── run.py                        # Entry point for QAMNIST analysis
│   └── rl
│       ├── train.py                          # Training code for RL environments
│       ├── utils.py                          # RL-specific helper functions
│       ├── plotting.py                       # Plotting utils specific to this task
│       ├── envs.py                           # Custom RL environment wrappers
│       ├── scripts/
│       │   ├── 4rooms/
│       │   │   └── *.sh                      # Training scripts for MiniGrid-FourRooms-v0 environment
│       │   ├── acrobot/
│       │   │   └── *.sh                      # Training scripts for Acrobot-v1 environment
│       │   └── cartpole/
│       │       └── *.sh                      # Training scripts for CartPole-v1 environment
│       └── analysis/
│           └── run.py                        # Entry point for RL analysis
├── data                                      # This is where data will be saved and downloaded to
│   └── custom_datasets.py                    # Custom datasets (e.g., Mazes), sort
├── models
│   ├── ctm.py                                # Main model code, used for: image classification, solving mazes, sort
│   ├── ctm_*.py                              # Other model code, standalone adjustments for other tasks
│   ├── ff.py                                 # feed-forward (simple) baseline code (e.g., for image classification)
│   ├── lstm.py                               # LSTM baseline code (e.g., for image classification)
│   ├── lstm_*.py                              # Other baseline code, standalone adjustments for other tasks
│   ├── modules.py                            # Helper modules, including Neuron-level models and the Synapse UNET
│   ├── utils.py                              # Helper functions (e.g., synch decay)
│   └── resnet.py                             # Wrapper for ResNet featuriser
├── utils
│   ├── housekeeping.py                       # Helper functions for keeping things neat
│   ├── losses.py                             # Loss functions for various tasks (mostly with reshaping stuff)
│   └── schedulers.py                         # Helper wrappers for learning rate schedulers
└── checkpoints
    └── imagenet, mazes, ...                  # Checkpoint directories (see google drive link for files)

```

## Setup
To set up the environment using conda:

```
conda create --name=ctm python=3.12
conda activate ctm
pip install -r requirements.txt
```

If there are issues with PyTorch versions, the following can be ran:
```
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Model training
Each task has its own (set of) training code. See for instance [tasks/image_classification/train.py](tasks/image_classification/train.py). We have set it up like this to ensure ease-of-use as opposed to clinical efficiency. This code is for researchers and we hope to have it shared in a way that fosters collaboration and learning. 

While we have provided reasonable defaults in the argparsers of each training setup, scripts to replicate the setups in the paper will typically be found in the accompanying script folders. If you simply want to dive in, run the following as a module (setup like this to make it easy to run many high-level training scripts from the top directory):

```
python -m tasks.image_classification.train
```
For debugging in VSCode, this configuration example might be helpful to you:
```
{
    "name": "Debug: train image classifier",
    "type": "debugpy",
    "request": "launch",
    "module": "tasks.image_classification.train",
    "console": "integratedTerminal",
    "justMyCode": false
}
```


## Running analyses

We also provide analysis and plotting code to replicate many of the plots in our paper. See `tasks/.../analysis/*` for more details on that. We also provide some data (e.g., the mazes we generated for training) and checkpoints (see [here](#checkpoints-and-data)). Note that ffmpeg is required for generating mp4 files from the analysis scripts. It can be installed with:
```
conda install -c conda-forge ffmpeg
```


## Checkpoints and data
You can download the data and checkpoints from here: 
- checkpoints: https://drive.google.com/drive/folders/1vSg8T7FqP-guMDk1LU7_jZaQtXFP9sZg
- maze data: https://drive.google.com/file/d/1cBgqhaUUtsrll8-o2VY42hPpyBcfFv86/view?usp=drivesdk

Checkpoints go in the `checkpoints` folder. For instance, when properly populated, the checkpoints folder will have the maze checkpoint in `checkpoints/mazes/...`

## UDL Rating Framework Integration

The UDL Rating Framework integrates with CTM through the `udl_rating_framework/models/ctm_adapter.py` module. This adapter enables neural approximation for fast UDL quality inference.

### Integration Architecture

The CTM adapter provides:

1. **UDLRatingCTM**: A PyTorch module that adapts CTM for UDL quality prediction
   - Token embedding layer for UDL text processing
   - Integration with `ContinuousThoughtMachine` from `models/ctm.py`
   - Rating head that maps synchronization to quality scores in [0,1]

2. **UDLTokenVocabulary**: Vocabulary management for UDL tokens
   - Maps tokens to integer indices for embedding lookup
   - Supports special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`

3. **TrackingData**: Container for CTM internal representations
   - Pre/post activations, synchronization data, attention weights
   - HDF5 serialization for analysis and visualization
   - Activation statistics and synchronization evolution metrics

### Usage Example

```python
from udl_rating_framework.models.ctm_adapter import UDLRatingCTM, UDLTokenVocabulary, create_udl_rating_model
from udl_rating_framework.core.representation import UDLRepresentation

# Create vocabulary and model
vocab = UDLTokenVocabulary()
udl = UDLRepresentation(udl_content, "example.udl")
vocab.add_tokens_from_udl(udl)

model = create_udl_rating_model(vocab_size=len(vocab))

# Tokenize and rate
token_ids = model.tokenize_udl(udl, vocab)
ratings, certainties, synch_out, _ = model(token_ids.unsqueeze(0))
print(f"Quality rating: {ratings.item():.3f}")
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Core dimensionality of CTM latent space |
| `d_input` | 64 | Dimensionality of projected attention outputs |
| `iterations` | 20 | Number of internal 'thought' ticks |
| `n_synch_out` | 32 | Number of neurons for output synchronization |
| `heads` | 8 | Number of attention heads |
| `memory_length` | 10 | History length for Neuron-Level Models |

### Mathematical Definition

Given a UDL represented as token sequence (t₁, t₂, ..., tₙ):

1. **Embed**: xᵢ = E(tᵢ) ∈ ℝᵈ
2. **Process**: S(T) = CTM(x₁, x₂, ..., xₙ)
3. **Rate**: q = σ(W·S(T) + b) ∈ [0,1]

The model leverages CTM's temporal processing and neural synchronization to produce quality ratings with associated certainty scores.
