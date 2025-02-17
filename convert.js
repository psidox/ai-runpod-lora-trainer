#!/usr/bin/env node
/**
 * This script scans a given LoRA data directory for .txt files that contain
 * keyword–delimiter–command pairs and converts them into a single TOML configuration
 * file with one section per text file.
 *
 * Usage:
 *   node convert_to_toml.js /path/to/lora_data -o training_config.toml -d ":"
 */

const fs = require('fs');
const path = require('path');
const { program } = require('commander');
const TOML = require('@iarna/toml');

// Try to convert a string value to an int, float, or boolean if possible.
function tryConvert(value) {
  const lowerVal = value.toLowerCase();
  if (lowerVal === 'true') return true;
  if (lowerVal === 'false') return false;

  // Check if it's a valid number
  if (!isNaN(value) && value.trim() !== '') {
    // If it doesn't contain a dot, try to convert to an integer.
    if (!value.includes('.')) {
      const intVal = parseInt(value, 10);
      if (String(intVal) === value.trim()) {
        return intVal;
      }
    }
    // Otherwise, convert to a float.
    const floatVal = parseFloat(value);
    return floatVal;
  }
  return value;
}

// Parse a single file into an object using the provided delimiter.
function parseFile(filepath, delimiter = ':') {
  const config = {};
  const content = fs.readFileSync(filepath, 'utf-8');

  // Split the file into lines (works for both Unix and Windows line endings)
  const lines = content.split(/\r?\n/);
  lines.forEach((line) => {
    line = line.trim();
    if (!line || line.startsWith('#')) {
      return; // Skip empty lines and comments
    }
    const index = line.indexOf(delimiter);
    if (index !== -1) {
      const key = line.substring(0, index).trim();
      const value = line.substring(index + delimiter.length).trim();
      config[key] = tryConvert(value);
    } else {
      console.warn(`Warning: Line in ${filepath} does not contain delimiter "${delimiter}": ${line}`);
    }
  });
  return config;
}

function main() {
  program
    .argument('<directory>', 'Directory containing the LoRA .txt files')
    .option('-o, --output <output>', 'Output TOML file', 'training_config.toml')
    .option('-d, --delimiter <delimiter>', 'Delimiter used in the .txt files', ':')
    .parse(process.argv);

  const options = program.opts();
  const loraDir = program.args[0];
  const outputFile = options.output;
  const delimiter = options.delimiter;

  // Ensure the directory exists.
  if (!fs.existsSync(loraDir) || !fs.statSync(loraDir).isDirectory()) {
    console.error(`Error: Directory "${loraDir}" does not exist or is not a directory.`);
    process.exit(1);
  }

  // Read the directory and filter for .txt files.
  const files = fs.readdirSync(loraDir);
  const txtFiles = files.filter((file) => file.toLowerCase().endsWith('.txt'));

  if (txtFiles.length === 0) {
    console.error(`No .txt files found in directory ${loraDir}`);
    process.exit(1);
  }

  // Aggregate configurations from each file.
  const aggregatedConfig = {};
  txtFiles.forEach((file) => {
    const filepath = path.join(loraDir, file);
    const sectionName = path.parse(file).name;
    const fileConfig = parseFile(filepath, delimiter);
    if (Object.keys(fileConfig).length > 0) {
      aggregatedConfig[sectionName] = fileConfig;
    } else {
      console.warn(`Warning: No valid key/value pairs found in ${filepath}`);
    }
  });

  // Convert the aggregated configuration object to a TOML string.
  const tomlString = TOML.stringify(aggregatedConfig);

  // Write the TOML string to the output file.
  try {
    fs.writeFileSync(outputFile, tomlString, 'utf-8');
    console.log(`Successfully wrote TOML configuration to ${outputFile}`);
  } catch (err) {
    console.error(`Error writing to ${outputFile}: ${err.message}`);
    process.exit(1);
  }
}

main();