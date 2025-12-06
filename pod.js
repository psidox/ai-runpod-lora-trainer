import { program } from "commander";
import { ApolloClient, InMemoryCache, HttpLink, gql } from "@apollo/client";
import { Client } from "ssh2";
import { Client as ScpClient } from "node-scp";
import { execSync } from "child_process";
import fs from "fs-extra";
import os from "os";
import path from "path";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";

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
  .option("--ostrisBaseImage <string>", "Tag for the Ostris base image")
  .option("--sdScriptsBaseImage <string>", "Tag for the sd-scripts base image")
  .option("--builtImageName <string>", "Tag for the prebuilt image that contains data and dependencies")
  .option("--baseImage <string>", "Base CUDA/PyTorch image used when building the custom image")
  .option("--pushBuiltImage", "Push the built image to the configured registry after build")
  .option("--awsAccessKeyId <string>", "AWS access key for S3 uploads")
  .option("--awsSecretAccessKey <string>", "AWS secret key for S3 uploads")
  .option("--s3Region <string>", "AWS region for S3")
  .option("--s3Bucket <string>", "S3 bucket to upload training artifacts")
  .option("--s3OutputPrefix <string>", "Prefix within the bucket for trained model uploads")
  .option("--s3ImageBucket <string>", "S3 bucket to store the prebuilt image tarball")
  .option("--s3ImageKey <string>", "S3 object key for the prebuilt image tarball")
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
  ostrisBaseImage: "ai-lora-ostris-base:latest",
  sdScriptsBaseImage: "ai-lora-sd-base:latest",
  builtImageName: "ai-lora-trainer:latest",
  baseImage: "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
  pushBuiltImage: false,
  awsAccessKeyId: "",
  awsSecretAccessKey: "",
  s3Region: "us-east-1",
  s3Bucket: "",
  s3OutputPrefix: "lora-outputs",
  s3ImageBucket: "",
  s3ImageKey: "lora-training-image.tar",
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
  buildTrainingImage(config)
    .then(() => process.exit(0))
    .catch((error) => {
      console.error("❌ Failed to build training image:", error);
      process.exit(1);
    });
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
  const entrypointPath = path.join(process.cwd(), "docker", "entrypoint.sh");
  if (!fs.existsSync(entrypointPath)) {
    throw new Error("docker/entrypoint.sh is missing. Please ensure the docker folder is present.");
  }
  return fs.readFileSync(entrypointPath, "utf8");
}

function generateDockerfile(config) {
  const datasetConfigName = path.basename(config.trainingConfigFile);
  const sdConfigName = config.sdScriptsConfigFile
    ? path.basename(config.sdScriptsConfigFile)
    : datasetConfigName;
  const baseModelName = getModelFilename(config.modelPath);
  return `FROM ${config.ostrisBaseImage || config.baseImage}

WORKDIR /workspace
RUN apt-get update && apt-get install -y git awscli wget && rm -rf /var/lib/apt/lists/*

# sd-scripts backend in addition to Ostris base
RUN rm -rf /workspace/${config.sdScriptsDirName} && \
    git clone ${config.sdScriptsRepo} /workspace/${config.sdScriptsDirName} && \
    pip install -r /workspace/${config.sdScriptsDirName}/requirements.txt && \
    pip install accelerate

# Copy dataset and configs
COPY dataset ${config.remoteDatasetPath}
COPY training-config ${config.remoteDatasetPath}/training-config

# Download base model into the image
RUN mkdir -p ${config.remoteModelsPath} && \
    wget -O ${config.remoteModelsPath}/${baseModelName} ${config.modelPath}

COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh
ENV TRAINING_BACKEND=${config.trainingBackend}
ENV TRAINING_CONFIG_PATH=${config.remoteDatasetPath}/training-config/${datasetConfigName}
ENV SD_SCRIPTS_CONFIG=${config.remoteDatasetPath}/training-config/${sdConfigName}
ENV OUTPUT_DIR=${config.trainOutputDir}
ENV MODEL_PATH=${config.remoteModelsPath}/${baseModelName}
ENV TOOLKIT_PRESET=${config.toolkitPreset}
ENV NETWORK_TYPE=z-image-turbo
ENV KEEP_ALIVE=${config.keepContainerAlive ? 1 : 0}
ENV S3_BUCKET=${config.s3Bucket}
ENV S3_OUTPUT_PREFIX=${config.s3OutputPrefix}
ENV AWS_DEFAULT_REGION=${config.s3Region}
CMD ["/workspace/entrypoint.sh"]
`;
}

function createS3Client(config) {
  return new S3Client({
    region: config.s3Region,
    credentials:
      config.awsAccessKeyId && config.awsSecretAccessKey
        ? {
            accessKeyId: config.awsAccessKeyId,
            secretAccessKey: config.awsSecretAccessKey,
          }
        : undefined,
  });
}

async function uploadFileToS3(filePath, bucket, key, config) {
  if (!bucket || !key) {
    console.log("ℹ️  Skipping S3 upload because bucket or key was not provided.");
    return;
  }
  const client = createS3Client(config);
  const body = fs.createReadStream(filePath);
  const fileSize = fs.statSync(filePath).size;
  console.log(`☁️  Uploading ${filePath} to s3://${bucket}/${key}...`);
  await client.send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: body,
      ContentLength: fileSize,
    })
  );
  console.log("✅ Image tarball uploaded to S3.");
}

function buildBaseImages(config) {
  console.log("🧱 Building Ostris base image...");
  runLocalCommand(
    `docker build -f docker/Dockerfile.ostris-base --build-arg BASE_IMAGE=${config.baseImage} -t ${config.ostrisBaseImage} .`
  );
  console.log("🧱 Building sd-scripts base image...");
  runLocalCommand(
    `docker build -f docker/Dockerfile.sd-scripts-base --build-arg BASE_IMAGE=${config.baseImage} -t ${config.sdScriptsBaseImage} .`
  );
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

async function buildTrainingImage(config) {
  console.log("📦 Building Docker image with datasets and dependencies included...");
  buildBaseImages(config);
  const contextDir = prepareBuildContext(config);
  runLocalCommand(`docker build -t ${config.builtImageName} ${contextDir}`);
  if (config.pushBuiltImage) {
    console.log(`🚀 Pushing image ${config.builtImageName} to registry...`);
    runLocalCommand(`docker push ${config.builtImageName}`);
  }

  const tarPath = path.join(os.tmpdir(), `lora-image-${Date.now()}.tar`);
  runLocalCommand(`docker save -o ${tarPath} ${config.builtImageName}`);
  const bucket = config.s3ImageBucket || config.s3Bucket;
  await uploadFileToS3(tarPath, bucket, config.s3ImageKey, config);
  fs.removeSync(tarPath);
  console.log(`✅ Image ready and archived: ${config.builtImageName}`);
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
  console.log("🚀 Launching training from the S3-archived container image...");
  const bucket = config.s3ImageBucket || config.s3Bucket;
  if (!bucket || !config.s3ImageKey) {
    throw new Error("S3 image bucket and key are required to retrieve the training image inside the pod.");
  }

  const installer = [
    "apt-get update",
    "apt-get install -y awscli docker.io",
    "service docker start || true"
  ].join(" && ");
  await executeCommand(ssh, installer, config);

  const awsEnvExports = [
    config.awsAccessKeyId ? `export AWS_ACCESS_KEY_ID='${config.awsAccessKeyId}'` : null,
    config.awsSecretAccessKey ? `export AWS_SECRET_ACCESS_KEY='${config.awsSecretAccessKey}'` : null,
    `export AWS_DEFAULT_REGION='${config.s3Region}'`,
  ]
    .filter(Boolean)
    .join(" && ");

  const downloadCommand = [awsEnvExports, `aws s3 cp s3://${bucket}/${config.s3ImageKey} /workspace/training-image.tar --region ${config.s3Region}`]
    .filter(Boolean)
    .join(" && ");
  await executeCommand(ssh, downloadCommand, config);

  const sdConfigName = config.sdScriptsConfigFile
    ? path.basename(config.sdScriptsConfigFile)
    : path.basename(config.trainingConfigFile);
  const runtimeEnv = [
    `-e TRAINING_BACKEND=${config.trainingBackend}`,
    `-e TRAINING_CONFIG_PATH=${config.remoteDatasetPath}/training-config/${path.basename(config.trainingConfigFile)}`,
    `-e SD_SCRIPTS_CONFIG=${config.remoteDatasetPath}/training-config/${sdConfigName}`,
    `-e OUTPUT_DIR=${config.trainOutputDir}`,
    `-e MODEL_PATH=${config.remoteModelsPath}/${getModelFilename(config.modelPath)}`,
    `-e TOOLKIT_PRESET=${config.toolkitPreset}`,
    `-e NETWORK_TYPE=z-image-turbo`,
    `-e KEEP_ALIVE=${config.keepContainerAlive ? 1 : 0}`,
    config.s3Bucket ? `-e S3_BUCKET=${config.s3Bucket}` : null,
    `-e S3_OUTPUT_PREFIX=${config.s3OutputPrefix}`,
    `-e AWS_DEFAULT_REGION=${config.s3Region}`,
    config.awsAccessKeyId ? `-e AWS_ACCESS_KEY_ID=${config.awsAccessKeyId}` : null,
    config.awsSecretAccessKey ? `-e AWS_SECRET_ACCESS_KEY=${config.awsSecretAccessKey}` : null,
  ]
    .filter(Boolean)
    .join(" ");

  const volumeFlags = [`-v ${config.trainOutputDir}:${config.trainOutputDir}`];

  const dockerCommands = [
    awsEnvExports,
    `mkdir -p ${config.trainOutputDir}`,
    "docker load -i /workspace/training-image.tar",
    `docker run --rm --gpus all --ipc=host --network=host ${volumeFlags.join(" ")} ${runtimeEnv} ${config.builtImageName}`,
  ]
    .filter(Boolean)
    .join(" && ");

  await executeCommand(ssh, dockerCommands, config);
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

  console.log("📦 Using S3-archived prebuilt image; skipping dataset/model uploads and dependency installs.");

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

if (action !== "build-image") {
  main(config);
}
