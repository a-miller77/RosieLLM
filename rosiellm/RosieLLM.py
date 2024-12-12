from rosiellm.RosieSSH import RosieSSH
from rosiellm.RosieJob import JobManager
from openai import OpenAI
from typing import Literal, Union
import random
import requests
import logging

MANAGEMENT_NODES = ['dh-mgmt1', 'dh-mgmt2', 'dh-mgmt3', 'dh-mgmt4']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RosieLLM:    
    def __init__(self,
                 job_name: str = 'RosieLLM',
                #  model: str = ..., #TODO trickle this down to other objects
                 rosie_username: str = None,
                 management_node: Literal[
                                    'dh-mgmt1',
                                    'dh-mgmt2',
                                    'dh-mgmt3',
                                    'dh-mgmt4'
                                ] = None,
                 return_openai_client: bool = True
                 ) -> Union['RosieLLM', OpenAI]:
        if not management_node or management_node not in MANAGEMENT_NODES:
            # Choose a random management node
            management_node = random.choice(MANAGEMENT_NODES)
        self.rosie_ssh_address = f"{management_node}.hpc.msoe.edu"
        # NOTE: RosieSSH assumes the address can be provided from .env which isn't compatible here
        self.rosie_ssh = RosieSSH(rosie_username, self.rosie_ssh_address)
        self.manager = JobManager(job_name, self.rosie_ssh)
        self.user = self.manager.user
        self.rosie_auth = self.manager.rosie_ssh.rosie_auth

        self.manager.launch_vllm_server()

        vllm_route = self.manager.BASE_URL.format(node_url=self.manager.node_url, port=self.manager.PORT) #TODO use a getter
        self.rosie_web_path = f"https://dh-ood.hpc.msoe.edu{vllm_route}"
        self.isRunning = False

        self._http_client = OpenAI(
            api_key="Not used",
            base_url=f"{self.rosie_web_path}/v1",
            default_headers={
                'Authorization': f'Basic {self.rosie_auth.get_rosie_auth()}',
                #NOTE: swap to FastAPI forwarder for AUTH at a later date
                'X-Authorization': f'Bearer {self.manager.token}'
            }
        )
        self._is_client_only = return_openai_client # Return only the client if requested

    @property
    def http_client(self):
        self.check_server_health()
        if self.isRunning:
            return self._http_client
        else:
            raise ConnectionError("Server is not running.")
        
    #NOTE: Should there be a setter?

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
                    logger.warning("Server cannot be accessed.")
            except requests.exceptions.RequestException as e:
                self.isRunning = False
                logger.error(f"Health check failed, Server not running: {e}")
            except Exception as e:
                self.isRunning = False
                logger.error(f"Unexpected error during health check: {e}")

    def __getattr__(self, name):
        # if _is_client_only is True, return attributes of the http_client when requested
        if self._is_client_only:
            return getattr(self.http_client, name)
        else:
            raise AttributeError(f"'RosieLLM' object has no attribute '{name}'")

    def __del__(self):
        self.manager.__del__()
