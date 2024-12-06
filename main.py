from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import (AgentExecutor, create_structured_chat_agent)
from langchain import hub
from langchain_core.messages import SystemMessage
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, date
import streamlit as st 
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import subprocess

# Set up API key and model
LLM_api = "AIzaSyDvbke4TODM1nOMbkZAXXhOVGQeECSsATU"
model = GoogleGenerativeAI(
    model='gemini-1.5-pro', 
    google_api_key=LLM_api, 
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
)

# Set up memory to retain conversation context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define system role and add initial message to memory
role = """
Throughout the entire session, strictly abide by these rules:
>>> we are using an Ubuntu Linux system and have installed nothing so if some command needs to be installed, install it first.
>>>You are a Linux Terminal command execution agent.
>>>Your only task is to execute the Linux Terminal commands provided to you.
>>>Do not perform any additional actions or make any assumptions about the commands.
>>>If the input is not a valid Linux Terminal command, respond with an error message asking for a proper Linux Terminal command.
>>>Do not provide explanations, reasoning, or details about the commands.
>>>Do not modify, interpret, or alter the provided commands in any way.
>>>No extra steps or behavior should be added—execute only what is asked.
>>>Your output should be only the result of the command execution or a message indicating invalid input.
>>>If the users enters an incorrect command, tell them to correct it with a suggestion
>>>Keep track of the current directory that you working in, and update it if a command asks you to change directory
"""
memory.chat_memory.add_message(SystemMessage(content=role))

# Initialize the current directory
current_directory = "/"

# Define a tool for command execution
@tool
def execute_command(command):
    """
    Executes a command on the command line and returns the output or error.
    
    Args:
        command (str): The command to be executed.
    
    Returns:
        str: The output or error from executing the command.
    """
    global current_directory

    # Check if the command is a change directory command (cd)
    if command.startswith('cd '):
        # Extract the directory to change to
        new_dir = command[3:].strip()

        # Handle relative paths by joining with the current directory
        if not os.path.isabs(new_dir):
            new_dir = os.path.join(current_directory, new_dir)

        # Check if the directory exists
        if os.path.isdir(new_dir):
            current_directory = new_dir
            return f"Changed directory to {current_directory}"
        else:
            return f"Error: The directory {new_dir} does not exist."
    
    # For other commands, execute them in the current directory
    try:
        result = subprocess.run(command, shell=True, check=True, cwd=current_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return f"Command failed with error:\n{e.stderr.decode('utf-8')}"

# Define a tool to list all folders starting from the root directory
@tool
def find_all_folders():
    """
    Navigates from the root directory and finds all the folders in the system.
    
    Returns:
        str: A list of all folder paths.
    """
    folders = []
    try:
        # Walk through the filesystem starting from the root
        for dirpath, dirnames, filenames in os.walk('/'):
            for dirname in dirnames:
                folders.append(os.path.join(dirpath, dirname))
        
        return '\n'.join(folders) if folders else "No folders found."
    except PermissionError:
        return "Error: Permission denied when accessing some directories."
    except Exception as e:
        return f"Error: {str(e)}"

# List of tools
tools = [execute_command, find_all_folders]

# Create an agent and executor
agent = create_structured_chat_agent(
    llm=model,
    tools=tools,
    prompt=hub.pull('hwchase17/structured-chat-agent'),
    stop_sequence=True
)

executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    max_execution_time=4000000,
    handle_parsing_errors=True
)

# Start loop to retain memory and keep asking user for queries
while True:
    user_input = input("USER NEEDS: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Ending session.")
        break
    response = executor.invoke({'input': user_input})
    print(response["output"])
