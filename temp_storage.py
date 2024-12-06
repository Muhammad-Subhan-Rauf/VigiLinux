from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import (AgentExecutor, create_structured_chat_agent)
from langchain import hub
from langchain_core.messages import SystemMessage
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json
from datetime import datetime, date
import streamlit as st 
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import subprocess
import time

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
User will tell you in natural language what they want. Figure out the necessary steps and commands to fulfill their requests and execute them with the given tool.
"""
memory.chat_memory.add_message(SystemMessage(content=role))

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
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return f"Command failed with error:\n{e.stderr.decode('utf-8')}"

# Define a tool for web scraping using Selenium
@tool
def scrape_web(url, selector):
    """
    Scrapes the web page at the specified URL for elements matching a CSS selector using Selenium.
    
    Args:
        url (str): The URL to scrape.
        selector (str): The CSS selector to search for (e.g., '.content-class', 'p').
    
    Returns:
        str: The text content of the elements found or an error message.
    """
    try:
        # Set up Selenium with headless Chrome
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        
        # Initialize the WebDriver (ensure 'chromedriver' is in your PATH or specify its path here)
        driver = webdriver.Chrome(options=options)
        
        # Navigate to the URL
        driver.get(url)
        
        # Wait for the content to load
        time.sleep(2)  # Adjust sleep time as necessary
        
        # Locate elements using the CSS selector
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        text_content = "\n".join([element.text for element in elements])
        
        # Close the driver
        driver.quit()
        
        return text_content if text_content else "No elements found with the specified selector."
    except Exception as e:
        return f"An error occurred while scraping the page: {e}"

tools = [execute_command, scrape_web]

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
    print("AGENT RESPONSE:", response)
