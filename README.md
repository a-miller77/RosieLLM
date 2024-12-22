# RosieLLM
RosieLLM is a Python library designed to streamline access to language models using the Rosie supercomputer's GPUS at MSOE. This library provides an intuitive interface, implementing OpenAI’s API. This enables students and faculty to efficiently utilize Rosie’s computational resources for running large-scale language models.

### Why RosieLLM?
Rosie is a high-performance computing (HPC) system designed primarily for large-scale, resource-intensive computational tasks. While it excels at tasks requiring large amounts of memory or parallel processing (e.g., scientific simulations and data analysis), HPC systems are not inherently user-friendly or designed for traditional development. RosieLLM allows a user to create and run applications in a standard local environment, while simultaneously taking advantage of Rosie's compute.

This means that students, faculty, and researchers at MSOE can focus on building apps and conducting experiments, without the headaches of HPC workflows, empowering users to build and research with LLM's in a simple and efficient way.

### Disclaimer
RosieLLM is currently in early beta. This means that drastic changes can happen, error checking is not particularly robust, and you can definitely prevent the system from running by tweaking arguments (see below). For safe, guaranteed results, stick with the default parameters. Additionally, this library is currently only supported on Windows. It will not work on Mac or locally on Rosie as of now.

## Features
- **Automatic SSH Tunneling**: Securely connects to ROSIE using SSH credentials to initialize sbatch jobs.
- **Automatic Job Creation**: Simplifies job creation and interaction with ROSIE.
- **Encypted Password Management**: The users Rosie credentials are safely stored, using both salting and encryption.
- **OpenAI Compatibility**: Mirrors OpenAI’s API structure for seamless integration with existing applications.
  - Token Streaming
  - Sync and Async modes
  - Integration into any workflow that is compatible with an OpenAI client
- **Modifiable Sbatch and vLLM Server Arguments**: Can manage a variety of args used to launch the vLLM server and Sbatch job (see 'Job Configuration' below)

## Installation

To install RosieLLM, use pip:

```bash
pip install git+https://github.com/a-miller77/RosieLLM.git
```

The first time you run the library, packages will automatically be installed on Rosie. This takes a few minutes.

## Usage

### Basic Example

The example below shows how to use RosieLLM to set up a job and interact with a language model.

```python
from rosiellm import RosieLLM

# Initialize RosieLLM
client = RosieLLM(
    job_name="RosieLLM",
    rosie_username="your_username",
)
# this will prompt a pop-up asking for your password. Note that the job spinning up takes around a minute on average.

# Interact with the model as an OpenAI client
completion = client.chat.completions.create(
    model=client.model,
    messages=[
        {"role": "user", "content": "What is AI Club at MSOE?"},
    ],
)

print(completion.choices[0].message.content)
```

## Job Configuration

The RosieLLM supports certain keyword arguments to modify the job submission. The following are all valid options:

### Accepted `kwargs` for RosieLLM
#### SLURM arguments
- **`job_name`**: The name of the job. (Default: `'RosieLLM'`)
- **`partition`**: The SLURM partition to be used. (Default: `'teaching'`)
- **`nodes`**: The number of nodes to allocate. (Default: `1`)
- **`gpus`**: The number of GPUs to allocate per node. (Default: `2`)
- **`cpus_per_gpu`**: The number of CPUs to allocate per GPU. (Default: `2`)
- **`out_file`**: The path to the job's output file. (Default: `/data/ai_club/RosieLLM/out/{user}_out.txt`)
- **`days`**: Days allocated for the job. (Default: `0`)
- **`hours`**: Hours allocated for the job. (Default: `3`)
- **`minutes`**: Minutes allocated for the job. (Default: `0`)
- **`container`**: Path to the Singularity container used for the job. (Default: `/data/ai_club/RosieLLM/RosieLLM.sif`). Not recommended to change.

If you don't recognize these, likely don't change them. See [ROSIE SLURM Details](https://msoe.dev/#/cli/sbatch) for more details.

#### vLLM arguments
- **`model`**: The HuggingFace model identifier to use. (Default: `'NousResearch/Meta-Llama-3-8B-Instruct'`)
- **`dtype`**: Data type precision, such as `half` for 16-bit precision. (Default: `'half'`)
- **`max_model_len`**: Maximum sequence length for the model. (Default: `2048`)
- **`download_dir`**: Directory to store downloaded models. (Default: `/data/ai_club/RosieLLM/models`)
- **`host`**: Host address for the job's server. (Default: `'0.0.0.0'`)
- **`port`**: Port for the job's server. (Default: dynamically set, e.g., `1234`)
- **`api_key`**: API key for authenticating requests. (Default: dynamically generated token)
- **`middleware`**: Middleware configuration for FastAPI. (Default: `'proxy_middleware.proxy_authentication'`). Not recommended to change.
- **`vllm_base_url`**: Base URL for the vLLM server. (Constructed dynamically using node URL and port). Not recommended to change.

If you don't recognize these, likely don't change them. See [vLLM server arguments](https://docs.vllm.ai/en/v0.6.0/serving/openai_compatible_server.html#command-line-arguments-for-the-server) for more details.

### Notes
1. If a key in `kwargs` matches an entry in the internal configuration dictionary, its value will override the default.
2. Valid entrys are NOT checked - change at your own risk.
3. Invalid keys in `kwargs` are ignored with a warning.
4. You can use these parameters to fine-tune resource allocation, job duration, and other runtime options.

## Roadmap

- **Version 0.1.0**: Focus on robust SSH tunneling, OpenAI compatibility, vLLM runs on Rosie.
- **Version 0.1.1**: Allow SLURM and server arguments to be set and modified printing/logging. Allow Async OpenAI client.
- **Future Enhancements**: Add support for auto-scaling, faster load times.
- **Future Overhaul**: Eventually the code will be rewritten to go through an API Gateway that can manage multiple models, authentication, and other features.

## Contributing

Due to the current stage of the project, please reach out to me over teams before writing any contributions. 

## License

RosieLLM is licensed under the Apache 2.0 License. See `LICENSE` for details.