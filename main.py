from openai import OpenAI
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
import os
from datasets import Dataset
import pandas as pd
from io import StringIO

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

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

dataset_str = (get_response("You're a movie critic", "Generate 50 movie reviews and tell me if its negative (0) or positive (1) and output that in a csv format where a row looks like: <review>, <0 or 1> (Don't forget the column names)"))
df = pd.read_csv(StringIO(dataset_str))

dataset = Dataset.from_pandas(df)


trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
)

trainer.train()