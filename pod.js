import { program } from "commander";
import { ApolloClient, InMemoryCache, HttpLink, gql } from "@apollo/client";
import { Client } from "ssh2";
import { Client as ScpClient } from "node-scp";
import { execSync } from "child_process";
import fs from "fs-extra";
import os from "os";
import path from "path";

// ---------------------------------------------------------------------
// Command-Line Arguments Setup Using Commander
// ---------------------------------------------------------------------
program
  .name("runpod_lora_train")
  .description("Deploy a pod on RunPod and train a LoRA model.")
  .argument("[action]", 'Action to perform, must be "run" to execute the training process')
  .option("--config <file>", "Path to configuration JSON file")
  .option("--minMemoryRequired <number>", "Minimum GPU memory required in GB", parseInt)
  .option("--minBidPriceLimit <number>", "Minimum bid price for spot instances", parseFloat)
  .option("--maxBidPriceLimit <number>", "Maximum bid price for spot instances", parseFloat)
  .option("--runpodApiKey <string>", "RunPod API key")
  .option("--instanceImage <string>", "Instance image to use")
  .option("--modelPath <string>", "Hugging Face model path")
  .option("--localDatasetPath <string>", "Local dataset directory")
  .option("--localOutputDir <string>", "Local output directory")
  .option("--trainingBackend <string>", "Training backend to invoke: ostris or sd-scripts")
  .option("--sdScriptsRepo <string>", "Repository for sd-scripts")
  .option("--sdScriptsDirName <string>", "Directory name for sd-scripts in the image")
  .option("--sdScriptsConfigFile <string>", "Config file to use with sd-scripts training")
  .option("--builtImageName <string>", "Tag for the prebuilt image that contains data and dependencies")
  .option("--baseImage <string>", "Base CUDA/PyTorch image used when building the custom image")
  .option("--pushBuiltImage", "Push the built image to the configured registry after build")
  .option("--keepContainerAlive", "Keep container running after training completes for debugging")
  .option("--debug", "Enable debug mode to log API, SSH, and SCP commands")
  .parse(process.argv);

const options = program.opts();
const action = program.args[0];

// ---------------------------------------------------------------------
// Default Configuration Values
// ---------------------------------------------------------------------
const defaultConfig = {
  volumeMountPath: "/workspace",
  remoteDatasetPath: "/workspace/dataset",
  trainOutputDir: "/workspace/lora_output",
  remoteModelsPath: "/workspace/models",
  runpodApiKey: "your-runpod-api-key", // Replace with your key or override via CLI/config.
  instanceImage: "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04",
  modelPath: "runwayml/stable-diffusion-v1-5",
  localDatasetPath: "./dataset",
  localOutputDir: "./output",
  toolkitRepo: "https://github.com/ostris/ai-toolkit.git",
  toolkitDirName: "ai-toolkit",
  toolkitPreset: "z-image-turbo",
  trainingConfigFile: "config.toml",
  trainingBackend: "ostris",
  sdScriptsRepo: "https://github.com/kohya-ss/sd-scripts.git",
  sdScriptsDirName: "sd-scripts",
  sdScriptsConfigFile: "sd-config.json",
  builtImageName: "ai-lora-trainer:latest",
  baseImage: "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
  pushBuiltImage: false,
  keepContainerAlive: false,
  minMemoryRequired: 15,      // in GB
  minBidPriceLimit: 0.1,       // Example: $0.10
  maxBidPriceLimit: 0.2,       // Example: $0.20
  debug: false
};

// ---------------------------------------------------------------------
// Load configuration from external JSON file if specified and merge with defaults
// ---------------------------------------------------------------------
let fileConfig = null;
if (options.config) {
  try {
    fileConfig = JSON.parse(fs.readFileSync(options.config, "utf8"));
    console.log(`Loaded configuration from ${options.config}`);
  } catch (error) {
    console.error("Failed to load configuration file:", error);
    process.exit(1);
  }
} else {
  try {
    fileConfig = fs.readFileSync('config.json', "utf8");
  } catch (error) {
    console.log(`No configuration filed found.`);
  }
  if (fileConfig) {
    try {
      fileConfig = JSON.parse(fileConfig);
      console.log(`Loaded configuration from config.json`);
    } catch (error) {
      console.error("Failed to parse config file:", error);
      process.exit(1);
    }
  }
}
const config = { ...defaultConfig, ...(fileConfig || {}), ...options };

// ---------------------------------------------------------------------
// If "run" is not passed as the first argument, print config and exit.
// ---------------------------------------------------------------------
if (!action || (action !== "run" && action !== "build-image")) {
  console.log("Current configuration options:");
  console.log(JSON.stringify(config, null, 2));
  console.log("\nTo run the script, pass 'run' as the first argument or 'build-image' to produce a preloaded image. For example:");
  console.log("  node runpod_lora_train.js build-image --config config.json");
  console.log("  node runpod_lora_train.js run --config config.json");
  process.exit(0);
}

if (action === "build-image") {
  buildTrainingImage(config);
  process.exit(0);
}

// ---------------------------------------------------------------------
// Helper function: debugGraphQLRequest
// ---------------------------------------------------------------------
async function debugGraphQLRequest(client, query, variables, config) {
  if (config.debug) {
    console.log("----- GraphQL Request -----");
    console.log("Query:", query);
    console.log("Variables:", variables);
  }
  try {
    const response = await client.query({ query, variables, fetchPolicy: "no-cache" });    
    if (config.debug) {
      console.log("----- GraphQL Response -----");
      console.log(JSON.stringify(response, null, 2));
    }
    return response;
  } catch (error) {
    if (config.debug) {
      console.error("----- GraphQL Error -----");
      console.error(JSON.stringify(error, null, 2));
    }
    throw error;
  }
}

// ---------------------------------------------------------------------
// FUNCTION DEFINITIONS (All functions receive parameters)
// ---------------------------------------------------------------------

/**
 * Extract a deterministic filename for the remote model download.
 */
function getModelFilename(modelPath) {
  if (!modelPath) return "model.safetensors";
  const cleaned = modelPath.replace(/\/$/, "");
  const parts = cleaned.split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "model.safetensors";
}

function runLocalCommand(command) {
  console.log(`$ ${command}`);
  execSync(command, { stdio: "inherit" });
}

/**
 * chooseGpuType(minMemory, minBidPriceLimit, maxBidPriceLimit, graphqlClient, config)
 *
 * Queries available GPU types and returns the chosen GPU type object that meets:
 *   - At least minMemory (in GB)
 *   - Has available capacity (totalCount > rentedCount)
 *   - Has a spot-instance bid price within the given range.
 * Exits if no GPU meets the criteria.
 */
async function chooseGpuType(minMemory, minBidPriceLimit, maxBidPriceLimit, graphqlClient, config) {
  const GPU_TYPES_QUERY = gql`
    query GpuTypes($input: GpuLowestPriceInput) {
      gpuTypes {
        id
        displayName
        memoryInGb
        lowestPrice(input: $input) {
          minimumBidPrice
          totalCount
          rentedCount
          uninterruptablePrice
        }
      }
    }
  `;
  try {
    const { data } = await debugGraphQLRequest(graphqlClient, GPU_TYPES_QUERY, {
      input: {
        gpuCount: 1,
        supportPublicIp: true,
        secureCloud: false,
      }
    }, config);
    const availableTypes = data.gpuTypes;
    const filtered = availableTypes.filter((gpu) => {
      const price = gpu.lowestPrice && gpu.lowestPrice.uninterruptablePrice;
      return (
        gpu.memoryInGb >= minMemory &&
        gpu.lowestPrice &&
        gpu.lowestPrice.uninterruptablePrice &&
        gpu.lowestPrice.totalCount > gpu.lowestPrice.rentedCount &&
        price >= minBidPriceLimit &&
        price <= maxBidPriceLimit
      );
    });
    if (filtered.length === 0) {
      console.error(
        `❌ No GPU type found with at least ${minMemory}GB memory and spot bid price between ${minBidPriceLimit} and ${maxBidPriceLimit}.`
      );
      process.exit(1);
    }
    filtered.sort((a, b) => {
      const priceDiff = a.lowestPrice.uninterruptablePrice - b.lowestPrice.uninterruptablePrice;
      return priceDiff !== 0 ? priceDiff : a.memoryInGb - b.memoryInGb;
    });
    const chosen = filtered[0];
    console.log(`✅ Selected GPU type: ${chosen.displayName} (${chosen.memoryInGb}GB)`);
    console.log(
      `   Spot Price: ${chosen.lowestPrice.minimumBidPrice} | Availability: ${chosen.lowestPrice.rentedCount}/${chosen.lowestPrice.totalCount}`
    );
    return chosen;
  } catch (error) {
    console.error("❌ Error querying GPU types:", error);
    process.exit(1);
  }
}

function generateEntrypointScript(config) {
  const datasetConfigName = path.basename(config.trainingConfigFile);
  const sdConfigName = config.sdScriptsConfigFile
    ? path.basename(config.sdScriptsConfigFile)
    : datasetConfigName;
  return `#!/bin/bash\nset -euo pipefail\n\nDATA_CONFIG="${config.remoteDatasetPath}/training-config/${datasetConfigName}"\nSD_CONFIG="${config.remoteDatasetPath}/training-config/${sdConfigName}"\nOUTPUT_DIR="${config.trainOutputDir}"\nMODEL_PATH="${config.remoteModelsPath}/${getModelFilename(config.modelPath)}"\nTRAINING_BACKEND="${config.trainingBackend}"\nTOOLKIT_PRESET="${config.toolkitPreset}"\nNETWORK_TYPE="z-image-turbo"\nKEEP_ALIVE="${config.keepContainerAlive ? 1 : 0}"\n\nif [ ! -d "$OUTPUT_DIR" ]; then\n  mkdir -p "$OUTPUT_DIR"\nfi\n\ncase "$TRAINING_BACKEND" in\n  sd-scripts)\n    echo "Running sd-scripts training using $SD_CONFIG"\n    accelerate launch /workspace/${config.sdScriptsDirName}/train_network.py --config "$SD_CONFIG"\n    ;;\n  ostris|*)\n    echo "Running Ostris AI Toolkit training using $DATA_CONFIG"\n    cd /workspace/${config.toolkitDirName}\n    python -m aitoolkit.train_lora \\\n      --preset "$TOOLKIT_PRESET" \\\n      --dataset_config "$DATA_CONFIG" \\\n      --model "$MODEL_PATH" \\\n      --output_dir "$OUTPUT_DIR" \\\n      --network_type "$NETWORK_TYPE"\n    ;;\nesac\n\nif [ "$KEEP_ALIVE" = "1" ]; then\n  echo "Training complete. Keeping container alive for inspection."\n  tail -f /dev/null\nfi\n`;
}

function generateDockerfile(config) {
  const datasetConfigName = path.basename(config.trainingConfigFile);
  const sdConfigName = config.sdScriptsConfigFile
    ? path.basename(config.sdScriptsConfigFile)
    : datasetConfigName;
  return `FROM ${config.baseImage}\n\nWORKDIR /workspace\nRUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*\n\n# Clone and install Ostris AI Toolkit\nRUN rm -rf /workspace/${config.toolkitDirName} && \\\n    git clone ${config.toolkitRepo} /workspace/${config.toolkitDirName} && \\\n    pip install --upgrade pip && \\\n    pip install -r /workspace/${config.toolkitDirName}/requirements.txt && \\\n    pip install -e /workspace/${config.toolkitDirName}\n\n# Clone sd-scripts for alternate backend\nRUN rm -rf /workspace/${config.sdScriptsDirName} && \\\n    git clone ${config.sdScriptsRepo} /workspace/${config.sdScriptsDirName} && \\\n    pip install -r /workspace/${config.sdScriptsDirName}/requirements.txt || true\nRUN pip install accelerate\n\n# Copy dataset and configs\nCOPY dataset ${config.remoteDatasetPath}\nCOPY training-config ${config.remoteDatasetPath}/training-config\n\n# Download base model into the image\nRUN mkdir -p ${config.remoteModelsPath} && \\\n    wget -O ${config.remoteModelsPath}/${getModelFilename(config.modelPath)} ${config.modelPath}\n\nCOPY entrypoint.sh /workspace/entrypoint.sh\nRUN chmod +x /workspace/entrypoint.sh\nENV TRAINING_BACKEND=${config.trainingBackend}\nENV TRAINING_CONFIG_PATH=${config.remoteDatasetPath}/training-config/${datasetConfigName}\nENV SD_SCRIPTS_CONFIG=${config.remoteDatasetPath}/training-config/${sdConfigName}\nENV OUTPUT_DIR=${config.trainOutputDir}\nENV MODEL_PATH=${config.remoteModelsPath}/${getModelFilename(config.modelPath)}\nCMD ["/workspace/entrypoint.sh"]\n`;
}

function prepareBuildContext(config) {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "lora-build-"));
  const datasetTarget = path.join(tempDir, "dataset");
  const configTarget = path.join(tempDir, "training-config");
  fs.ensureDirSync(datasetTarget);
  fs.ensureDirSync(configTarget);

  if (!fs.existsSync(config.localDatasetPath)) {
    throw new Error(`Dataset path not found: ${config.localDatasetPath}`);
  }
  fs.copySync(config.localDatasetPath, datasetTarget);

  if (!fs.existsSync(config.trainingConfigFile)) {
    throw new Error(`Training config file not found: ${config.trainingConfigFile}`);
  }
  fs.copyFileSync(config.trainingConfigFile, path.join(configTarget, path.basename(config.trainingConfigFile)));
  if (config.sdScriptsConfigFile && fs.existsSync(config.sdScriptsConfigFile)) {
    fs.copyFileSync(config.sdScriptsConfigFile, path.join(configTarget, path.basename(config.sdScriptsConfigFile)));
  }

  fs.writeFileSync(path.join(tempDir, "entrypoint.sh"), generateEntrypointScript(config), { mode: 0o755 });
  fs.writeFileSync(path.join(tempDir, "Dockerfile"), generateDockerfile(config));
  return tempDir;
}

function buildTrainingImage(config) {
  console.log("📦 Building Docker image with datasets and dependencies included...");
  const contextDir = prepareBuildContext(config);
  runLocalCommand(`docker build -t ${config.builtImageName} ${contextDir}`);
  if (config.pushBuiltImage) {
    console.log(`🚀 Pushing image ${config.builtImageName} to registry...`);
    runLocalCommand(`docker push ${config.builtImageName}`);
  }
  console.log(`✅ Image ready: ${config.builtImageName}`);
}

/**
 * startRunPodInstance(graphqlClient, config, gpuTypeId)
 *
 * Deploys a new pod using a GraphQL mutation. Returns the pod ID.
 */
async function startRunPodInstance(graphqlClient, config, gpuType) {
  const POD_FIND_AND_DEPLOY_MUTATION = gql`
    mutation OnDemand($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        imageName
        env
        machineId
        machine {
          podHostId
        }
      }
    }
  `;
  const variables = {
    input: {
      cloudType: "COMMUNITY",
      gpuCount: 1,
      volumeInGb: 0,
      volumeKey: null,
      containerDiskInGb: 40,
      minVcpuCount: 2,
      minMemoryInGb: 15,
      gpuTypeId: gpuType.id,
      name: "SD-Scripts",
      imageName: config.instanceImage,
      ports: "22/tcp",
      startSsh: true,
      volumeMountPath: config.volumeMountPath,
    },
  };
  try {
    const { data } = await debugGraphQLRequest(graphqlClient, POD_FIND_AND_DEPLOY_MUTATION, variables, config);
    const podId = data.podFindAndDeployOnDemand.id;
    console.log(`🎉 RunPod instance started! ID: ${podId}`);
    return podId;
  } catch (error) {
    console.error("❌ Error starting RunPod instance:", error);
    process.exit(1);
  }
}

/**
 * startRentPodInstance(graphqlClient, config, gpuTypeId)
 *
 * Deploys a new pod using a GraphQL mutation. Returns the pod ID.
 */
async function startRentPodInstance(graphqlClient, config, gpuType) {
  // console.log(gpuType)
  const POD_FIND_AND_DEPLOY_MUTATION = gql`
    mutation BidPod($input: PodRentInterruptableInput!) {
      podRentInterruptable(input: $input) {
        id
        imageName
        env
        machineId
        machine {
          podHostId
        }
      }
    }
  `;
  const variables = {
    input: {
      cloudType: "ALL",
      gpuCount: 1,
      volumeInGb: 0,
      volumeKey: null,
      containerDiskInGb: 40,
      minVcpuCount: 2,
      minMemoryInGb: 15,
      gpuTypeId: gpuType.id,
      name: "RunPod Tensorflow",
      imageName: config.instanceImage,
      ports: "22/tcp",
      startSsh: true,
      volumeMountPath: config.volumeMountPath,
      bidPerGpu: gpuType.lowestPrice.minimumBidPrice + 0.05,
    },
  };
  try {
    const { data } = await debugGraphQLRequest(graphqlClient, POD_FIND_AND_DEPLOY_MUTATION, variables, config);
    const podId = data.podFindAndDeployOnDemand.id;
    console.log(`🎉 RunPod instance started! ID: ${podId}`);
    return podId;
  } catch (error) {
    console.error("❌ Error starting RunPod instance:", error);
    process.exit(1);
  }
}


/**
 * waitForPodReady(graphqlClient, podId, config)
 *
 * Polls the pod status until it is ready and returns the pod IP.
 */
async function waitForPodReady(graphqlClient, podId, config) {
  console.log("⏳ Waiting for instance to be ready...");
  const POD_QUERY = gql`
    query Pod($podId: String!) {
      pod(input: { podId: $podId }) {
        id
        name
        runtime {
          uptimeInSeconds
          ports {
            ip
            isIpPublic
            privatePort
            publicPort
          }
        }
      }
    }
  `;
  while (true) {
    try {
      const variables = { podId };
      const { data } = await debugGraphQLRequest(graphqlClient, POD_QUERY, variables, config);
      if (data.pod && data.pod.runtime && data.pod.runtime.ports && data.pod.runtime.ports.length > 0) {
        const instance =
          data.pod.runtime.ports.find((p) => p.isIpPublic) || data.pod.runtime.ports[0];
        console.log(`✅ RunPod instance is ready! IP: ${instance.ip} Port: ${instance.publicPort}`);
        return instance;
      }
    } catch (error) {
      console.error("Error checking pod status:", error);
    }
    await new Promise((resolve) => setTimeout(resolve, 10000));
  }
}

/**
 * stopRunPodInstance(graphqlClient, podId, config)
 *
 * Stops the pod using a GraphQL mutation.
 */
async function stopRunPodInstance(graphqlClient, podId, config) {
  const POD_STOP_MUTATION = gql`
    mutation PodStop($input: PodStopInput!) {
      podStop(input: $input) {
        id
        desiredStatus
      }
    }
  `;
  const variables = { input: { podId } };
  try {
    const { data } = await debugGraphQLRequest(graphqlClient, POD_STOP_MUTATION, variables, config);
    if (data.podStop && data.podStop.desiredStatus) {
      console.log(`🛑 RunPod instance ${podId} stopped!`);
    } else {
      console.error("❌ Failed to stop RunPod instance.");
    }
  } catch (error) {
    console.error("❌ Error stopping RunPod instance:", error);
  }
}

/**
 * connectSSH(podIp, username)
 *
 * Establishes and returns an SSH connection to the given pod IP.
 */
async function connectSSH(instance, username) {
  return new Promise((resolve, reject) => {
    const ssh = new Client();
    ssh
      .on("ready", () => {
        console.log("🔗 SSH connection established.");
        resolve(ssh);
      })
      .on("error", (err) => {
        console.error("❌ SSH connection failed:", err);
        reject(err);
      })
      .connect({
        host: instance.ip,
        port: instance.publicPort,
        username,
        privateKey: fs.readFileSync('/Users/josh/.ssh/id_rsa', 'utf8'),
      });
  });
}

/**
 * uploadDataset(localDatasetPath, instance, username, remoteDatasetPath, config)
 *
 * Uploads the local dataset directory to the pod.
 */
async function uploadDataset(localDatasetPath, instance, username, remoteDatasetPath, config) {
  console.log("📤 Uploading dataset...");
  if (config.debug) {
    console.log(`DEBUG: Executing SCP upload: ${localDatasetPath} -> ${username}@${instance.ip}:${remoteDatasetPath}`);
  }
  try {
    const client = await new ScpClient({
      host: instance.ip,
      port: instance.publicPort,
      username,
      privateKey: fs.readFileSync('/Users/josh/.ssh/id_rsa', 'utf8'),
    });
    await client.uploadDir(localDatasetPath, remoteDatasetPath);
    console.log("✅ Dataset uploaded!");
  } catch (error) {
    console.error("❌ Dataset upload failed:", error);
    process.exit(1);
  }
}

/**
 * executeCommand(ssh, command, config)
 *
 * Executes a command over the given SSH connection.
 * In debug mode, logs the command before executing it.
 */
async function executeCommand(ssh, command, config) {
  if (config.debug) {
    console.log("DEBUG: Executing SSH command:", command);
  }
  return new Promise((resolve, reject) => {
    ssh.exec(command, (err, stream) => {
      if (err) {
        console.error("❌ SSH command failed:", err);
        return reject(err);
      }
      stream
        .on("close", () => {
          if (config.debug) {
            console.log("DEBUG: SSH command completed.");
          }
          resolve();
        })
        .on("data", (data) => {
          console.log(data.toString());
        })
        .stderr.on("data", (data) => {
          console.error(data.toString());
        });
    });
  });
}

/**
 * cloneAiToolkit(ssh, config, toolkitPath)
 *
 * Clones the Ostris AI Toolkit repository into the workspace.
 */
async function cloneAiToolkit(ssh, config, toolkitPath) {
  console.log("📥 Cloning Ostris AI Toolkit repository...");
  const command = `rm -rf ${toolkitPath} && git clone ${config.toolkitRepo} ${toolkitPath}`;
  await executeCommand(ssh, command, config);
}

/**
 * installPythonRequirements(ssh, config, toolkitPath)
 *
 * Installs the Python requirements for the Ostris AI Toolkit repository.
 */
async function installPythonRequirements(ssh, config, toolkitPath) {
  console.log("📦 Installing Python requirements for Ostris AI Toolkit...");
  const command = `cd ${toolkitPath} &&
    pip install --upgrade pip &&
    pip install --upgrade -r requirements.txt &&
    pip install -e .
`;
  await executeCommand(ssh, command, config);
}

/**
 * launchTraining(ssh, toolkitPath, datasetConfigPath, trainOutputDir, baseModelPath, config)
 *
 * Launches the LoRA training process using the Ostris AI Toolkit repository.
 */
async function launchTraining(
  ssh,
  config
) {
  console.log("🚀 Launching training inside the prebuilt container image...");
  const datasetConfigName = path.basename(config.trainingConfigFile);
  const sdConfigName = config.sdScriptsConfigFile
    ? path.basename(config.sdScriptsConfigFile)
    : datasetConfigName;
  const env = [
    `TRAINING_BACKEND=${config.trainingBackend}`,
    `OUTPUT_DIR=${config.trainOutputDir}`,
    `MODEL_PATH=${config.remoteModelsPath}/${getModelFilename(config.modelPath)}`,
    `TRAINING_CONFIG_PATH=${config.remoteDatasetPath}/training-config/${datasetConfigName}`,
    `SD_SCRIPTS_CONFIG=${config.remoteDatasetPath}/training-config/${sdConfigName}`,
    `TOOLKIT_PRESET=${config.toolkitPreset}`,
    `KEEP_ALIVE=${config.keepContainerAlive ? 1 : 0}`,
  ].join(" ");
  const trainingCommand = `${env} /workspace/entrypoint.sh`;
  await executeCommand(ssh, trainingCommand, config);
}

/**
 * downloadModel(ssh, remoteModelsPath, modelPath, config)
 *
 * Downloads the base model from Hugging Face into the remote models directory.
 */
async function downloadModel(ssh, config, targetModelPath) {
  console.log("📥 Downloading model from Hugging Face...");
  const command = `
    mkdir -p ${config.trainOutputDir} &&
    mkdir -p ${config.remoteModelsPath} &&
    wget -q -O ${targetModelPath} ${config.modelPath}
  `;
  await executeCommand(ssh, command, config);
}

/**
 * downloadOutput(instance, username, trainOutputDir, localOutputDir, config)
 *
 * Downloads the training output from the pod to a local directory.
 */
async function downloadOutput(instance, username, trainOutputDir, localOutputDir, config) {
  console.log("📥 Downloading output...");
  if (config.debug) {
    console.log(`DEBUG: Executing SCP download: ${username}@${instance.ip}:${trainOutputDir} -> ${localOutputDir}`);
  }
  try {
    const client = await new ScpClient({
      host: instance.ip,
      port: instance.publicPort,
      username,
      privateKey: fs.readFileSync('/Users/josh/.ssh/id_rsa', 'utf8'),
    });
    await client.downloadDir(trainOutputDir, localOutputDir);
    console.log("✅ Output downloaded!");
  } catch (error) {
    console.error("❌ Output download failed:", error);
    process.exit(1);
  }
}

// ---------------------------------------------------------------------
// MAIN SCRIPT FLOW
// ---------------------------------------------------------------------
async function main(config) {
  // Ensure the runtime image points to the prebuilt artifact when provided.
  config.instanceImage = config.builtImageName || config.instanceImage;

  // Create a GraphQL client using the provided API key.
  const graphqlClient = new ApolloClient({
    link: new HttpLink({
      uri: "https://api.runpod.io/graphql",
      fetch: fetch,
      useGETForQueries: false,
      headers: {
        Authorization: `Bearer ${config.runpodApiKey}`,
        "Content-Type": "application/json"
      },
    }),
    cache: new InMemoryCache(),
  });

  // 0. Choose a GPU type that meets the criteria.
  const chosenGpu = await chooseGpuType(
    config.minMemoryRequired,
    config.minBidPriceLimit,
    config.maxBidPriceLimit,
    graphqlClient,
    config
  );
  
  // 1. Deploy a new pod.
  const podId = await startRunPodInstance(graphqlClient, config, chosenGpu);
  // const podId = "c49ssi55fz1lwj";

  // 2. Wait until the pod is ready and get its IP.
  const instance = await waitForPodReady(graphqlClient, podId, config);

  // 3. Establish an SSH connection (using username "root").
  const sshConnection = await connectSSH(instance, "root");

  console.log("📦 Using prebuilt image; skipping dataset/model uploads and dependency installs.");

  // Launch the training directly from the image contents.
  await launchTraining(sshConnection, config);

  // Download the training output.
  await downloadOutput(instance, "root", config.trainOutputDir, config.localOutputDir, config);

  // 10. Stop the pod.
  await stopRunPodInstance(graphqlClient, podId, config);

  // Close the SSH connection.
  sshConnection.end();
  console.log("✅ Process complete!");
  process.exit(0);
}

main(config);
