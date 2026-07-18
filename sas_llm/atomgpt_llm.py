import agapi #MUST DOWNLOAD VERSION agapi==2025.11.15
from openai import AsyncOpenAI, OpenAI
from agents import function_tool, Agent, OpenAIChatCompletionsModel
from agents import set_tracing_disabled, Runner, ModelSettings
from agapi.client import Agapi
import numpy as np
import re
import asyncio
import os
import sys
from datetime import datetime


_SAS_LLM_DIR = os.path.dirname(os.path.abspath(__file__))


async def use_llm(api_key, model, input_text, save_dir):
    # if mode == 'weighted_sum':
    #     instructions_path = os.path.join(_SAS_LLM_DIR, "instructions_weighted_sum.txt")
    # elif mode == 'distribution':
    #     instructions_path = os.path.join(_SAS_LLM_DIR, "instructions_distribution.txt")
    # else:
    #     print('ERROR: Set mode to weighted_sum or distribution')
    #     sys.exit()

    instructions_path = os.path.join(_SAS_LLM_DIR, "instructions_weighted_sum.txt")
    with open(instructions_path, "r", encoding="utf-8") as f:
        instructions = f.read()

    set_tracing_disabled(disabled=True)

    client = AsyncOpenAI(
        base_url="https://atomgpt.org/api",
        api_key=api_key
    )

    agent = Agent(
        name="Assistant",
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model=model,
            openai_client=client
        )
    )

    result = await Runner.run(agent, input_text)

    code = extract_python_code(result.final_output)

    folder = create_datetime_folder(base_path=save_dir)

    with open(folder + '/generated_script.py', "w", encoding="utf-8") as f:
        f.write(code)

    with open(folder + '/user_input_text.txt', "w") as file:
        file.write(input_text)


def extract_python_code(text):
    """
    Extracts Python code from a string that may contain markdown fences.
    """
    # Remove triple backtick code fences
    code = re.sub(r"```python", "", text)
    code = re.sub(r"```", "", code)
    return code.strip()


async def run_llm(agent, input_text):
    result = await Runner.run(agent, input_text)
    return result

def create_datetime_folder(date_str=None, time_str=None, base_path="."):
    """
    Creates a folder named with date and time.

    Args:
        date_str (str): Date in 'YYYY-MM-DD' format (optional)
        time_str (str): Time in 'HH-MM-SS' format (optional)
        base_path (str): Where to create the folder (default = current directory)

    Returns:
        str: Path of the created folder
    """

    # If no date/time provided, use current datetime
    if date_str is None or time_str is None:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

    folder_name = f"{date_str}_{time_str}"
    folder_path = os.path.join(base_path, folder_name)

    os.makedirs(folder_path, exist_ok=True)

    return folder_path



def list_atomgpt_models(api_key):
    client = OpenAI(base_url="https://atomgpt.org/api", api_key=api_key)
    try:
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        print(f"Error listing models: {type(e).__name__}: {e}")
        return []

def print_atomgpt_models(api_key):
    for name in list_atomgpt_models(api_key):
        print(name)