from rosiellm.RosieSSH import RosieSSH
import tempfile
import time
import os
import textwrap
import secrets
import logging

from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobManager:
    def __init__(self, job_name='RosieLLM', rosie_ssh: RosieSSH = None, **kwargs):
        self.job_name = job_name.strip()
        if rosie_ssh:
            if not rosie_ssh.ssh_client:
                rosie_ssh.connect()
            self.rosie_ssh = rosie_ssh
        else:
            self.rosie_ssh = RosieSSH()
            self.rosie_ssh.connect()
        self.user = self.rosie_ssh.ssh_username
        self.token = secrets.token_urlsafe() #look into jwt(?)
        self.PORT = 1234 #TODO scan for open port
        self.node_url = None
        self.BASE_URL = "/node/{node_url}.hpc.msoe.edu/{port}"

        self.config_dict = {
            'job_name': job_name,
            'partition': 'teaching',
            'nodes': 1,
            'gpus': 2,
            'cpus_per_gpu': 2,
            'out_file': f'/data/ai_club/RosieLLM/out/{self.user}_out.txt',
            'days': 0,
            'hours': 3,
            'minutes': 0,
            'container': "/data/ai_club/RosieLLM/RosieLLM.sif",
            'model': "NousResearch/Meta-Llama-3-8B-Instruct",
            'dtype': "half",
            'max_model_len': "2048",
            'download_dir': "/data/ai_club/RosieLLM/models", #TODO: Change to tmp dir
            'host': "0.0.0.0",
            'port': str(self.PORT),
            'api_key': self.token,
            'middleware': 'proxy_middleware.proxy_authentication', #TODO: remove once proxy server implemented
            'vllm_base_url': self.BASE_URL.format(node_url="$SLURMD_NODENAME", port=self.PORT)
        }

        if kwargs:
            logger.warning("Providing additional arguments the the job manager might cause unexpected behavior.")
            for key, value in kwargs.items():
                if key in self.config_dict:
                    previous = self.config_dict[key]
                    self.config_dict[key] = value
                    logger.info(f"Updated {key} from {previous} > {value}")
                else:
                    logger.warning(f"Invalid argument (ignored): {'{'}{key}:{value}{'}'}")
    
    # def __del__(self):
        # self.rosie_ssh.execute_instance_command(f'scancel -n {self.job_name}')
        # self.rosie_ssh.__del__()

    def launch_vllm_server(self) -> None:
        """
        Launches the initial job on Rosie.
        """
        try:
            sbatch_script = self.create_llm_sbatch()
            local_script_path, remote_script_path = self.create_temp_sbatch_script(sbatch_script)

            logger.debug(f"Local SBATCH Script Path: {local_script_path}")
            logger.debug(f"Remote SBATCH Script Path: {remote_script_path}")

            # Copy the SBATCH script directly to the remote server
            self.rosie_ssh.copy_file_to_remote(local_script_path, remote_script_path)
            os.remove(local_script_path)
            self.rosie_ssh.execute_command(f'chmod +x {remote_script_path}')

            # Execute the sbatch command on the remote server
            #this current implementation causes the server to not be able to be shut down later
            self.rosie_ssh.execute_command(f'sbatch {remote_script_path}', streaming=True)
            # self.rosie_ssh.execute_command(f'rm {remote_script_path}')
            self.node_url = self.get_node_url(self.job_name)

        except Exception as e:
            #TODO: improve(?)
            logger.error(f"An error occurred: {e}")

    def create_temp_sbatch_script(self, sbatch_script: str) -> Tuple[str, str]:
        try:
            # Normalize line endings to Unix style NOTE: Will not work on MacOS
            sbatch_script_unix = sbatch_script.replace('\r\n', '\n').replace('\r', '\n')

            # Create a temporary SBATCH script locally with Unix line endings
            with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='\n', suffix='.sh') as temp_file:
                temp_file.write(sbatch_script_unix)
                temp_file.flush()
                local_script_path = temp_file.name

            # Define the remote script path
            remote_script_path = f'/data/ai_club/RosieLLM/{os.path.basename(local_script_path)}' 
            return local_script_path, remote_script_path

        except Exception as e:
            logger.error(f"Failed to create temporary SBATCH script: {e}")
            raise

    def get_node_url(self, job_name, timeout: int = 20) -> str:
        """
        Retrieves the URL of the node where a specific job is running, polling until the job is running.
        Args:
            job_name (str): The name of the job to check.
            timeout (int, optional): The maximum number of seconds to wait for the job to start. Defaults to 60.
        Returns:
            str: The URL of the node where the job is running, in the form "dh-nodeX" or "dh-nodeXX", 
            where X is the node.
        Credit to Jackson Rolando, Kevin Paganini, Jennifer Madigan, Nathan Cernik, Tyler Cernik.
        """
        get_job_url_command = f'squeue -u {self.user} -n {job_name} -o "%N %T"'
        node_url = None
        squeue_out = None
        
        start = time.time()
        while node_url is None and time.time() - start < timeout:
            # self.rosie_ssh.wait_for_ready_channel(timeout=timeout)
            squeue_out = self.rosie_ssh.execute_instance_command(get_job_url_command)
            if squeue_out and 'dh' in squeue_out and 'RUNNING' in squeue_out:
                current_jobs = squeue_out.strip().split('\n')[1:]
                node_url = current_jobs[-1].split(' ')[0]
            else:
                time.sleep(0.5)
        if node_url is None:
            raise TimeoutError(f"Failed to find job URL for job {job_name} in {timeout} seconds.")

        print(f"Job URL Found: {node_url}")
        return node_url

    def create_llm_sbatch(self) -> str:                
        cfg = self.config_dict
        time = f'{cfg["days"]}-{cfg["hours"]}:{cfg["minutes"]}:00'

        # Constructing the command using an f-string with curly braces around parameters
        vllm_command = (
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {cfg['model']} "
            f"--dtype {cfg['dtype']} "
            f"-tp {cfg['gpus']} "
            # f"--max-model-len {cfg['max_model_len']} " #is this necessary?
            f"--download-dir {cfg['download_dir']} "
            f"--host {cfg['host']} "
            f"--port {cfg['port']} "
            f"--root-path {cfg['vllm_base_url']} "
            f"--middleware {cfg['middleware']}"
            # f"--enable-cors "
            # f"--log-level INFO "
            # f"--num-workers 4 "
            # f"--max-batch-size 32 "
            # f"--metrics-port 9090 "
        )

        # Print the constructed command
        logger.debug(vllm_command)

        sbatch_script = textwrap.dedent(
f'''            #!/bin/bash
            #SBATCH --job-name='{cfg['job_name']}'
            #SBATCH --partition={cfg['partition']}
            #SBATCH --nodes={cfg['nodes']}
            #SBATCH --gpus={cfg['gpus']}
            #SBATCH --cpus-per-gpu={cfg['cpus_per_gpu']}
            #SBATCH --time={time}
            #SBATCH --output={cfg['out_file']}
            
            container="{cfg['container']}"
            
            singularity exec --nv -B /data:/data -B /data:/scratch/data ${{container}} bash -c '
            # Ensure Python dependencies are installed
            if ! python -c "import vllm" 2>/dev/null; then
                echo "vllm not found. Installing..."
                python -m pip install --user --upgrade pip vllm
            else
                echo "vllm already installed."
            fi &&
            
            # Change directory to the project root
            echo "Changing directory to project root..."
            echo "Directory: $(pwd) -> $(cd /data/ai_club/RosieLLM && pwd)"
            
            # Set environment variables
            export PYTHONPATH=/data/ai_club/RosieLLM:$PYTHONPATH &&
            export ROSIE_VLLM_API_KEY={self.token} &&
            echo "Added API_KEY to environment variables and updated PYTHONPATH"
            
            # Run the vLLM server
            {vllm_command}
            '
            '''
        )
        return sbatch_script