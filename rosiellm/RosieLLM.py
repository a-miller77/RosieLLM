from rosiellm.RosieSSH import RosieSSH
from rosiellm.RosieJob import JobManager
from openai import OpenAI, AsyncOpenAI
from typing import Literal, Union
import random
import requests
import logging

MANAGEMENT_NODES = ['dh-mgmt1', 'dh-mgmt2', 'dh-mgmt3', 'dh-mgmt4']
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

class RosieLLM:
    def __init__(self,
                 job_name: str = 'RosieLLM',
                 rosie_username: str = None,
                 management_node: Literal[
                                    'dh-mgmt1',
                                    'dh-mgmt2',
                                    'dh-mgmt3',
                                    'dh-mgmt4'
                                ] = None,
                 use_as_openai_client: bool = True,
                 async_client: bool = False,
                 log_level: Union[int, str] = logging.WARN,
                 **kwargs
                 ) -> 'RosieLLM':
        """
        Initialize a RosieLLM object.

        Args:
            job_name (str): The name of the job to be created on Rosie.
            model (str): The model to be used for the LLM. Must be a valid HuggingFace model name.
            rosie_username (str): The username to be used for authentication with Rosie.
            management_node (Literal: 'dh-mgmt[1,2,3,4]', optional): The management node to be used for the server. If not provided, a random node will be chosen.
            return_openai_client (bool): If True, the RosieLLM object can be used as if it were an OpenAI client.
            async_client (bool): If True, the OpenAI client will be asynchronous.
        """
        logger.setLevel(log_level)
        if not management_node or management_node not in MANAGEMENT_NODES:
            # Choose a random management node
            if management_node not in MANAGEMENT_NODES:
                logger.warning(f"Invalid management node provided, choosing a random node.")
            management_node = random.choice(MANAGEMENT_NODES)
        self.rosie_ssh_address = f"{management_node}.hpc.msoe.edu"
        # NOTE: RosieSSH assumes the address can be provided from .env which isn't compatible here
        self.rosie_ssh = RosieSSH(rosie_username, self.rosie_ssh_address)
        self.manager = JobManager(job_name, self.rosie_ssh, **kwargs)
        self.user = self.manager.user
        self.rosie_auth = self.manager.rosie_ssh.rosie_auth
        self.model = self.manager.config_dict['model']

        self.manager.launch_vllm_server()

        vllm_route = self.manager.BASE_URL.format(node_url=self.manager.node_url, port=self.manager.PORT) #TODO use a getter
        self.rosie_web_path = f"https://dh-ood.hpc.msoe.edu{vllm_route}"
        self.isRunning = False

        if not self.manager.node_url:
            logger.error("Server failed to launch.")
        else:
            print("Job has been launched, waiting for server to start (this can take over a minute)...")
            logger.info("See your job progress here:")
            logger.info(f"https://dh-ood.hpc.msoe.edu/pun/sys/dashboard/files/fs//data/ai_club/RosieLLM/out/{self.user}_out.txt")

        base_url = f"{self.rosie_web_path}/v1"
        default_headers = {
            'Authorization': f'Basic {self.rosie_auth.get_rosie_auth()}',
            #NOTE: swap to FastAPI forwarder for AUTH at a later date
            'X-Authorization': f'Bearer {self.manager.token}'
        }

        self._http_client = OpenAI(
            api_key="None", base_url=base_url, default_headers=default_headers
            ) if not async_client else AsyncOpenAI(
            api_key="None", base_url=base_url, default_headers=default_headers
            )
        self._is_client = use_as_openai_client # Return only the client if requested

    @property
    def http_client(self):
        self.check_server_health()
        if self.isRunning:
            return self._http_client
        else:
            raise ConnectionError("Server is not running. Server launch can be slow, try again in a moment.")
        
    @http_client.setter
    def http_client(self, value):
        logger.warning("http_client was designed to be read-only, override at your own risk.")
        self._http_client = value
        self.isRunning = False

    def check_server_health(self):
        if not self.isRunning:
            try:
                logger.info("Checking server health...")
                r = requests.get(f"{self.rosie_web_path}/health",
                                 headers={'Authorization': f'Basic {self.rosie_auth.get_rosie_auth()}'}
                )
                self.isRunning = r.status_code == 200
                logger.info(f"Health check status: {r.status_code}: {requests.status_codes._codes[r.status_code][0]}")
                if self.isRunning:
                    logger.info("Server is running.")
                else:
                    # if server can be reached but /health doesn't return 200, it's likely there is an Auth issue
                    logger.error("Server cannot be accessed.")
            except requests.exceptions.RequestException as e:
                self.isRunning = False
                logger.info(f"Health check failed, Server not running: {e}")
            except Exception as e:
                self.isRunning = False
                logger.critical(f"Unexpected error during health check: {e}")

    def __getattr__(self, name):
        # if _is_client is True, return attributes of the http_client when requested
        if self._is_client:
            return getattr(self.http_client, name)
        else:
            raise AttributeError(f"'RosieLLM' object has no attribute '{name}'")