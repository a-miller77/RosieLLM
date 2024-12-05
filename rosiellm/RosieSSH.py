import os
import re
import time
import uuid
import hashlib
import base64
from select import select
from threading import Lock
from getpass import getpass
from typing import Tuple, Optional

import paramiko
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()
USERNAME = os.getenv('USERNAME')
ADDRESS = os.getenv('ADDRESS')

class RosieSSH:
    """
    A class to manage SSH connections and execute commands on Rosie.
    """
    def __init__(self, ssh_username: str = None, ssh_host: str = None):
        """
        Initialize the SSH connection parameters.
        This method initializes the SSH connection parameters either from the provided arguments
        or from environment variables. If any of the required parameters (username, password, host)
        are missing, it raises a ValueError.
        Args:
            ssh_username (str, optional): The SSH username. Defaults to None.
            ssh_host (str, optional): The SSH host. Defaults to None.
        Raises:
            ValueError: If any of the SSH credentials (username, password, host) connect be loaded.
        """
        self.ssh_username = ssh_username or USERNAME
        self.ssh_host = ssh_host or ADDRESS
        self.rosie_auth = RosieAuth(self.ssh_username)
        
        if not self.ssh_username or not self.ssh_host:
            raise ValueError("""All SSH credentials (USERNAME, HOST) 
                             must be provided either as arguments or environment variables.""")
        # main client, used persistently
        self.ssh_client = None
        self.channel = None

        # used for individual commands
        self.instance_client = None

        self.lock = Lock()

    def __del__(self):
        self.close()

    def connect(self) -> None:
        """
        Establishes an SSH connection to Rosie.
        Raises:
            paramiko.SSHException: If there is any error while connecting to the remote server.
        """
        # Use paramiko to establish an SSH connection
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(self.ssh_host, username=self.ssh_username, password=self.rosie_auth.get_rosie_password())
            self.channel = self.ssh_client.invoke_shell()

            self.instance_client = paramiko.SSHClient()
            self.instance_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.instance_client.connect(self.ssh_host, username=self.ssh_username, password=self.rosie_auth.get_rosie_password())
        except paramiko.SSHException as e:
            self.close()
            raise paramiko.SSHException(f"An error occurred while connecting to {self.ssh_host}: {e}")

        # Flush the initial banner
        self.flush_buffer()

    def close(self) -> None:
        """
        Closes the SSH connection to Rosie.
        """
        if self.channel:
            self.channel.close()
            self.channel = None
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
        if self.instance_client:
            self.instance_client.close()
            self.instance_client = None

    def execute_instance_command(self, command: str) -> Optional[str]:
        """
        Executes a command on the remote server through a new SSH connection using Paramiko.
        Args:
            command (str): The command to be executed on the remote server.
        Returns:
            str: The output of the command execution.
        Raises:
            paramiko.SSHException: If there is any error while executing the command.
        """
        if not self.instance_client:
            raise paramiko.SSHException("SSH connection is not established. Call connect() method first.")
        
        _, stdout, stderr = self.instance_client.exec_command(command)
        output = stdout.read().decode(errors='ignore')
        error = stderr.read().decode(errors='ignore')
        return output + error
        
    def execute_command(self, command: str, streaming: bool = False, timeout=20) -> Optional[str]:
        """
        Executes a command on the remote server through an SSH tunnel using Paramiko.
        Args:
            command (str): The command to be executed on the remote server.
            streaming (bool, optional): If True, the output will be streamed as it arrives.
        Returns:
            str: The output of the command execution. If streaming is True, returns None.
        Raises:
            paramiko.SSHException: If the SSH connection is not established, 
                or if there is any error while executing the command.
        """
        with self.lock:
            if not self.channel:
                raise paramiko.SSHException("SSH connection is not established. Call connect() method first.")

            # Generate random 8-character hex codes
            start_code = uuid.uuid4().bytes[:4].hex()
            end_code = uuid.uuid4().bytes[:4].hex()
            # debug_buffer = ""

            # Send command
            command = f'echo "{start_code}" && {command} && echo "{end_code}"\n'
            self.channel.send(command)
            
            overflow = self.__flush_buffer_until_key(f'{end_code}"')
            # debug_buffer += (command + overflow)

            output = ["", overflow]  # Initialize output list with an empty string
            buffer_size = 32 if streaming else 4096
            prompt_detected = False
            
            while not prompt_detected:
                self.wait_for_ready_channel(timeout=timeout)
                if self.channel.recv_ready():
                    response = self.channel.recv(buffer_size).decode(errors='ignore')
                    last_response = output[-1]
                    buffer = "".join([last_response, response])
                    # debug_buffer += response
                    response_modified = False

                    if start_code in buffer:
                        buffer = buffer.split(start_code, 1)[1] # Remove everything before start_code
                        output[-1] = buffer # update the last response because it has been modified
                        response_modified = True
                    if end_code in buffer:
                        buffer = buffer.split(end_code, 1)[0] # Remove everything after end_code
                        output[-1] = buffer # update the last response because it has been modified
                        response_modified = True
                        prompt_detected = True

                    if streaming:
                        print(output[-2], end='')  # Print buffer as it comes
                    if not response_modified:
                        output.append(response)
            cleaned_output = ""
            if streaming:
                print(output[-1], end='')  # Print the last buffer
            else:
                cleaned_output = "".join(output)

                # Remove ANSI escape codes
                ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
                cleaned_output = ansi_escape.sub('', cleaned_output)
                cleaned_output = cleaned_output.replace('\r', '')

        overflow = self.flush_buffer()
        # print(f'{len(overflow)}: {overflow}')
        return cleaned_output if not streaming else "" #or debug) else debug_buffer 
    
    def copy_file_to_remote(self, local_file_path: str, rosie_file_path: str) -> None:
        """
        Copies the given text to a file on the remote server.
        Args:
            text (str): The text to be copied to the file.
            file_path (str): The path of the file on the remote server.
        """
        if not self.channel:
            raise paramiko.SSHException("SSH connection is not established. Call connect() method first.")
        sftp = self.ssh_client.open_sftp()
        sftp.put(local_file_path, rosie_file_path)
        sftp.close()
    
    def send_password(self, message: str = None) -> None:
        """
        Sends the password to the remote server.
        Args:
            password (str): The password to be sent to the remote server.
        """
        # self.channel.send(self.ssh_password + '\n')
        self.channel.send(self.rosie_auth.get_rosie_password() + '\n')
        self.flush_buffer()

    def cancel(self) -> None:
        """
        Cancels the current command execution on the remote server.
        """
        self.channel.send('\x03')

    def flush_buffer(self, num_bytes: int = 4096) -> str:
        """
        Empties the buffer of the SSH channel.
        """
        self.channel.send(" ") #guard against empty buffer
        self.wait_for_ready_channel()
        output = []
        while self.channel.recv_ready():
            output.append(self.channel.recv(num_bytes))
        return "".join([o.decode(errors='ignore') for o in output])[:-1]
    
    def __flush_buffer_until_key(self, key: str) -> str:
        """
        Clears the buffer of the SSH channel until the specified key is found, and returns the remaining buffer.
        """
        self.wait_for_ready_channel()
        output = [""]
        buffer_size = max(2*len(key), 32)
        key_found = False
        overflow = ""

        while not key_found:
            while self.channel.recv_ready() and not key_found:
                response = self.channel.recv(buffer_size).decode(errors='ignore')
                recent_buffer = "" + output[-1] + response
                output.append(response)
                output.pop(0)
                if key in recent_buffer:
                    overflow = recent_buffer.split(key, 1)[1]  # Remove everything before key
                    key_found = True
        return overflow
    
    def wait_for_ready_channel(self, timeout: int = 20) -> None:
        """
        Wait until the SSH channel is ready to read or until the timeout is reached.
        Args:
            timeout (int): Maximum time in seconds to wait for the channel to be ready.
        Raises:
            TimeoutError: If the channel is not ready within the given timeout.
        """
        start = time.time()
        while not self.__channel_is_ready():
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for the channel to be ready.")

    def __channel_is_ready(self, timeout: int = 1) -> bool:
        """
        Check if the SSH channel is ready to read.
        Args:
            timeout (int): Maximum time in seconds to wait for the channel to be ready.
        Returns:
            bool: True if the channel is ready, False otherwise.
        Raises:
            paramiko.SSHException: If the SSH channel is not established.
        """
        if not self.channel:
            raise paramiko.SSHException("SSH connection is not established. Call connect() method first.")
        
        ready_to_read, _, _ = select([self.channel], [], [], timeout)
        return self.channel in ready_to_read

class RosieAuth:
    """
    A class to securely manage Rosie authentication credentials using encryption.
    
    Attributes:
        auth_token (bytes): The encrypted base64 encoding of "username:password" for basic authentication.
        password (bytes): The encrypted password.
        cipher (Fernet): The Fernet cipher object used for encryption and decryption.
    
    Methods:
        get_rosie_auth() -> str:
            Decrypts and returns the base64-encoded "username:password" for basic authentication.
        get_rosie_password() -> str:
            Decrypts and returns the password.
    """
    def __init__(self, username: str, password: Optional[str] = None, iterations: int = 100000):
        """
        Initialize the RosieAuth with a username and an optional password.
        Args:
            username (str): The username for Rosie authentication.
            password (str, optional): The password to be encrypted and stored.
            iterations (int, optional): The number of iterations for key derivation. Defaults to 100000.
        """
        salt = os.urandom(16)
        encryption_key = base64.urlsafe_b64encode(hashlib.pbkdf2_hmac('sha256', username.encode(), salt, iterations))
        self.cipher = Fernet(encryption_key)

        self.auth_token = None
        self.password = None

        if password:
            self.__set_credentials(username, password)
        else:
            self.__set_credentials(username, getpass(f"Enter the Rosie Password for {username}: "))

    def __set_credentials(self, username: str, password: str) -> None:
        """
        Encrypts and sets the username and password for authentication.
        Args:
            username (str): The username for Rosie authentication.
            password (str): The password to be encrypted and stored.
        """
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        self.auth_token = self.cipher.encrypt(encoded_credentials.encode())
        self.password = self.cipher.encrypt(password.encode())

    def get_rosie_auth(self) -> str:
        """
        Decrypts and returns the base64-encoded "username:password" for basic authentication.
        Returns:
            str: The decrypted base64-encoded "username:password" string.
        Raises:
            ValueError: If the authentication token is not set.
        """
        if not self.auth_token:
            raise ValueError("Authentication token is not set.")
        return self.cipher.decrypt(self.auth_token).decode("utf-8")

    def get_rosie_password(self) -> str:
        """
        Decrypts and returns the password.
        Returns:
            str: The decrypted password.
        Raises:
            ValueError: If the password is not set.
        """
        if not self.password:
            raise ValueError("Password is not set.")
        return self.cipher.decrypt(self.password).decode()