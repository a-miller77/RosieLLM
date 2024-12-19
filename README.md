# RosieLLM

RosieLLM is a Python library designed to streamline access to language models on the ROSIE supercomputer at MSOE. This library provides an intuitive interface, similar to OpenAI’s API, enabling students and faculty to efficiently utilize ROSIE’s computational resources for running large-scale language models.

### Disclaimer
RosieLLM is currently in early beta - this means that drastic changes will happen, error checking is not particularly robust, and you can definitely prevent the system from running by tweaking kwargs. For safe, guaranteed results, stick with the default parameters.

## Features
- **Automatic SSH Tunneling**: Securely connects to ROSIE using SSH credentials to initialize sbatch jobs.
- **User-Friendly Interface**: Simplifies interaction with ROSIE, making it accessible for technical and non-technical users.
- **OpenAI Compatibility**: Mirrors OpenAI’s API structure for seamless integration with existing applications.
  - Token Streaming
  - Sync and Async modes
  - Integration into any workflow that is compatible with an OpenAI client
- **Sbatch and vLLM server kwargs**
  - Can manage a variety of args used to launch the vLLM server and Sbatch job (see 'Job Configuration' below)

## Why RosieLLM?

Rosie is great, but navigating the system and using its compute within other applications can be difficult. Having a standard way for a project not running on ROSIE to communicate with an LLM using Rosie's compute is advantageous in several scenarios.

## Installation

To install RosieLLM, use pip:

```bash
pip install git+https://github.com/a-miller77/RosieLLM.git
```

## Usage

### Basic Example

Here’s how to use RosieLLM to set up a job and interact with a language model:

```python
from rosiellm import RosieLLM

# Initialize RosieLLM
client = RosieLLM(
    job_name="RosieLLM",
    rosie_username="your_username",
)
# this will prompt a pop-up asking for your password. Your password is stored in a salted and encrypted state for the duration of the program.

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

### Accepted `kwargs`

- **`job_name`**: The name of the job. (Default: `'RosieLLM'`)
- **`partition`**: The SLURM partition to be used. (Default: `'teaching'`)
- **`nodes`**: The number of nodes to allocate. (Default: `1`)
- **`gpus`**: The number of GPUs to allocate per node. (Default: `2`)
- **`cpus_per_gpu`**: The number of CPUs to allocate per GPU. (Default: `2`)
- **`out_file`**: The path to the job's output file. (Default: `/data/ai_club/RosieLLM/out/{user}_out.txt`)
- **`days`**: Days allocated for the job. (Default: `0`)
- **`hours`**: Hours allocated for the job. (Default: `3`)
- **`minutes`**: Minutes allocated for the job. (Default: `0`)
- **`container`**: Path to the Singularity container used for the job. (Default: `/data/ai_club/RosieLLM/RosieLLM.sif`)
- **`model`**: The HuggingFace model identifier to use. (Default: `'NousResearch/Meta-Llama-3-8B-Instruct'`)
- **`dtype`**: Data type precision, such as `half` for 16-bit precision. (Default: `'half'`)
- **`max_model_len`**: Maximum sequence length for the model. (Default: `2048`)
- **`download_dir`**: Directory to store downloaded models. (Default: `/data/ai_club/RosieLLM/models`)
- **`host`**: Host address for the job's server. (Default: `'0.0.0.0'`)
- **`port`**: Port for the job's server. (Default: dynamically set, e.g., `1234`)
- **`api_key`**: API key for authenticating requests. (Default: dynamically generated token)
- **`middleware`**: Middleware configuration for FastAPI. (Default: `'proxy_middleware.proxy_authentication'`)
- **`vllm_base_url`**: Base URL for the vLLM server. (Constructed dynamically using node URL and port)

### Notes
1. If a key in `kwargs` matches an entry in the internal configuration dictionary, its value will override the default.
2. Valid entrys are NOT checked - change at your own risk.
3. Invalid keys in `kwargs` are ignored with a warning.
4. You can use these parameters to fine-tune resource allocation, job duration, and other runtime options.

## Roadmap

- **Version 0.1.0**: Focus on robust SSH tunneling, OpenAI compatibility, vLLM runs on Rosie.
- **Version 0.1.1**: Allow kwargs to be set and modified printing/logging. Allow Async OpenAI client.
- **Future Enhancements**: Add support for auto-scaling, faster load times.
- **Future Overhaul**: Eventually the code will be rewritten to go through an API Gateway that can manage multiple models, authentication, and other features.

## Contributing

Due to the current stage of the project, please reach out to me over teams before writing any contributions. 

## License

RosieLLM is licensed under the Apache 2.0 License. See `LICENSE` for details.