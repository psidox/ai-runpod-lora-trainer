# RunPod LoRA Training CLI

`runpod_lora_train.js` is a Node.js command-line tool that automates deploying a GPU pod on [RunPod](https://www.runpod.io/) to train a **LoRA** (Low-Rank Adaptation) model, using the [sd-scripts](https://github.com/kohya-ss/sd-scripts) repository.

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
3. **SSH + SCP**: Automatically transfers local dataset to the remote pod, then downloads training outputs.
4. **LoRA Training**: Automates installing dependencies and launching `sd-scripts` to train a LoRA.
5. **Debug Mode**: Logs GraphQL requests/responses, SSH commands, and other details.

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

```bash
node start run --config config.json
```

### CLI Arguments

- `[action]`: Must be **`run`** to start the process. If omitted, the script prints the current configuration and exits.
- `--config <file>`: Path to a JSON config file (merged with defaults).
- `--minMemoryRequired <number>`: Minimum GPU memory in GB.
- `--minBidPriceLimit <number>`: Minimum spot bid price.
- `--maxBidPriceLimit <number>`: Maximum spot bid price.
- `--runpodApiKey <string>`: Your RunPod API key.
- `--instanceImage <string>`: Docker image to use.
- `--modelPath <string>`: Hugging Face model path or direct URL.
- `--localDatasetPath <string>`: Local dataset directory.
- `--localOutputDir <string>`: Local output directory.
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
6. **Upload Dataset**: Uses SCP to send your local dataset to the instance.
7. **Download Model**: Pulls or downloads the base model to the remote environment.
8. **Clone & Install**: Clones `sd-scripts`, installs Python dependencies.
9. **Train LoRA**: Executes the training script via `accelerate launch ...`.
10. **Download Output**: Retrieves artifacts and logs.
11. **Stop Pod**: Shuts down the GPU instance to avoid further billing.

---

## Example

```bash
# 1. Prepare config.json
{
  "runpodApiKey": "YOUR_RUNPOD_API_KEY",
  "modelPath": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/model.ckpt",
  "localDatasetPath": "./dataset",
  "localOutputDir": "./output",
  "minMemoryRequired": 24,
  "minBidPriceLimit": 0.05,
  "maxBidPriceLimit": 0.2,
  "debug": true
}

# 2. Run
npm start run --config config.json
```

---

## Notes & Limitations

- **Private Key Path**: This script uses a hardcoded private key path (`~/.ssh/id_rsa`). Modify or generalize if needed.
- **Prices & Limits**: The script picks the first GPU that matches your memory/price constraints. Ensure your price range is realistic.
- **SSH Key**: You must have a valid SSH key on your local machine.
- **Interruptible Pods**: If you want to use a spot (interruptible) instance, you may need to modify the relevant GraphQL mutation.
