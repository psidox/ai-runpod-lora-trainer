# RunPod LoRA Training CLI

`runpod_lora_train.js` is a Node.js command-line tool that automates deploying a GPU pod on [RunPod](https://www.runpod.io/) to train a **LoRA** (Low-Rank Adaptation) model. It can build a custom Docker image that already contains your dataset, base model, and both [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit) and [sd-scripts](https://github.com/kohya-ss/sd-scripts) so that the pod only needs to pull and run the image.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Workflow Outline](#workflow-outline)
- [Example](#example)
- [Notes & Limitations](#notes--limitations)
- [License](#license)

---

## Features

1. **Command-Line Parsing**: Uses [Commander](https://www.npmjs.com/package/commander) for flexible CLI arguments.
2. **RunPod GraphQL**: Interacts with RunPod’s GraphQL API to:
   - Query available GPUs.
   - Start/stop GPU pods.
   - Monitor pod status.
3. **SSH + SCP**: Automatically transfers training outputs after the run.
4. **LoRA Training**: Automates installing dependencies and launching either the Ostris AI Toolkit (`z-image-turbo` preset) or sd-scripts from the same container image.
5. **Debug Mode**: Logs GraphQL requests/responses, SSH commands, and other details.
6. **Preloaded Images**: Build a Docker image with your dataset, configs, and base model baked in, then run pods directly from that image.

---

## Installation

1. **Clone** or **download** this repository (or place the script in your project).
2. **Install dependencies**:
   ```bash
   npm install
   ```

---

## Usage

Run:

1. Build the preloaded image (optional but recommended):

```bash
node pod.js build-image --config config.json
```

2. Run training on RunPod using that image:

```bash
node pod.js run --config config.json
```

### CLI Arguments

- `[action]`: Use **`build-image`** to bake your dataset/configs into a Docker image, or **`run`** to deploy and train from that image. If omitted, the script prints the current configuration and exits.
- `--config <file>`: Path to a JSON config file (merged with defaults).
- `--minMemoryRequired <number>`: Minimum GPU memory in GB.
- `--minBidPriceLimit <number>`: Minimum spot bid price.
- `--maxBidPriceLimit <number>`: Maximum spot bid price.
- `--runpodApiKey <string>`: Your RunPod API key.
- `--instanceImage <string>`: Docker image to use.
- `--modelPath <string>`: Hugging Face model path or direct URL.
- `--localDatasetPath <string>`: Local dataset directory.
- `--localOutputDir <string>`: Local output directory.
- `--trainingBackend <string>`: `ostris` (default) or `sd-scripts`.
- `--builtImageName <string>`: Tag for the custom image that contains dataset/configs.
- `--baseImage <string>`: Base CUDA/PyTorch image used when building the custom image.
- `--pushBuiltImage`: Push the built image to its registry.
- `--keepContainerAlive`: Keep the pod running after training for debugging.
- `--debug`: Enable verbose logging for GraphQL, SSH, and SCP.


---

## Configuration

The script merges:
1. **Defaults** (hardcoded in the script)
2. **`config.json`** (if found or specified via `--config`)
3. **CLI arguments**

### Example `config.json`
```json
{
  "runpodApiKey": "YOUR_RUNPOD_API_KEY",
  "modelPath": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/model.ckpt",
  "localDatasetPath": "./dataset",
  "localOutputDir": "./output",
  "trainingBackend": "ostris",
  "builtImageName": "your-registry/ai-lora-trainer:latest",
  "baseImage": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
  "minMemoryRequired": 15,
  "minBidPriceLimit": 0.1,
  "maxBidPriceLimit": 0.2,
  "debug": false
}
```

---

## Workflow Outline

1. **Check/parse config**: The script merges CLI arguments, your JSON config, and defaults.
2. **Choose GPU**: Queries RunPod GraphQL for a GPU that meets memory and price constraints.
3. **Deploy Pod**: Starts an on-demand GPU instance.
4. **Wait for Ready**: Polls RunPod until the instance is ready.
5. **SSH**: Connects to the instance as `root` via SSH.
6. **Train LoRA**: Runs `/workspace/entrypoint.sh` from the prebuilt image to trigger either Ostris AI Toolkit or sd-scripts training.
7. **Download Output**: Retrieves artifacts and logs.
8. **Stop Pod**: Shuts down the GPU instance to avoid further billing.

---

## Example

```bash
# 1. Prepare config.json
{
  "runpodApiKey": "YOUR_RUNPOD_API_KEY",
  "modelPath": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/model.ckpt",
  "localDatasetPath": "./dataset",
  "localOutputDir": "./output",
  "builtImageName": "your-registry/ai-lora-trainer:latest",
  "minMemoryRequired": 24,
  "minBidPriceLimit": 0.05,
  "maxBidPriceLimit": 0.2,
  "debug": true
}

# 2. Build the preloaded image
node pod.js build-image --config config.json

# 3. Run training on RunPod using that image
node pod.js run --config config.json
```

---

## Notes & Limitations

- **Private Key Path**: This script uses a hardcoded private key path (`~/.ssh/id_rsa`). Modify or generalize if needed.
- **Prices & Limits**: The script picks the first GPU that matches your memory/price constraints. Ensure your price range is realistic.
- **SSH Key**: You must have a valid SSH key on your local machine.
- **Interruptible Pods**: If you want to use a spot (interruptible) instance, you may need to modify the relevant GraphQL mutation.
