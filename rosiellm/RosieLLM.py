from rosiellm.RosieSSH import RosieSSH
from rosiellm.RosieJob import JobManager
from openai import OpenAI
from typing import Literal, Union
import random

MANAGEMENT_NODES = ['dh-mgmt1', 'dh-mgmt2', 'dh-mgmt3', 'dh-mgmt4']

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
                 return_openai_client: bool = True
                 ) -> Union['RosieLLM', OpenAI]:
        if not management_node or management_node not in MANAGEMENT_NODES:
            # Choose a random management node
            management_node = random.choice(MANAGEMENT_NODES)
        self.rosie_ssh_address = f"{management_node}.hpc.msoe.edu"
        # RosieSSH assumes the address can be provided from .env which isn't compatible here
        self.rosie_ssh = RosieSSH(rosie_username, self.rosie_ssh_address)
        self.manager = JobManager(job_name, self.rosie_ssh)
        self.user = self.manager.user
        self.rosie_auth = self.manager.rosie_ssh.rosie_auth

        self.manager.launch_vllm_server()

        vllm_route = self.manager.BASE_URL.format(node_url=self.manager.node_url, port=self.manager.PORT) #TODO use a getter
        self.rosie_web_path = f"https://dh-ood.hpc.msoe.edu{vllm_route}/v1"

        self.client = OpenAI(
            api_key=self.manager.token,
            base_url=self.rosie_web_path,
            default_headers={
                'Authorization': f'Basic {self.rosie_auth.get_rosie_auth()}'
            }
        )

        # Return only the client if requested
        if return_openai_client:
            self._is_client_only = True
        else:
            self._is_client_only = False

    def __getattr__(self, name):
        if self._is_client_only:
            return getattr(self.client, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __del__(self):
        self.manager.__del__()
