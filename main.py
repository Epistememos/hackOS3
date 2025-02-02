from openai import OpenAI
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
import os
from datasets import Dataset
import pandas as pd
from io import StringIO

load_dotenv()


client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


df = pd.read_csv('apache_error_parsed.csv').head(10)
csv_string = df.to_string(index=False)
training_args = SFTConfig(
    max_seq_length=512,
    output_dir="/tmp",
)

def get_response(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

dataset_str = (get_response("You're a senior developer, propose valid solutions for each error logs and link them up", csv_string))

print(dataset_str)
