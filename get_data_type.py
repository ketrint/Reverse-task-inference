import pandas as pd
import tqdm
import os
from os.path import join, dirname
from openai import OpenAI, APIError
import re
from dotenv import load_dotenv

# Loading env variables
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Load your data
code_blocks = pd.read_csv('./notebooks_meta.csv')

#openai_api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI(
    api_key = "YOUR_API_KEY"
)

# Defining prompts
prompt = "Сonstruct a dictionary named 'graph_inference', you could do it based on the relevant information produced by the code. This dictionary should contain Data and Transformation keys. The Data corresponds to the data types changings made by applying transformations. You MUST NOT include the parts of saving data, and for inference - applying NN or ML use ApplyNN (ApplyML) with type of model as a param. "

results_df = pd.DataFrame(columns=['kaggle_id', 'code', 'data_type', 'rating', 'comp_id'])

cnt = 5
for index, row in tqdm.tqdm(code_blocks.iterrows(), total=code_blocks.shape[0]):
    code = row['CleanCode']

    try:
        # Send the code to the GPT-4 API with the first prompt
        response1 = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user", 
                    "content":f'{code}\n Get the data type using in this code. Chose only one from: tabular, image, text, video, audio, time_series. You response MUST return ONLY one word of data type and nothing else!',
                },
            ]
        )
        result1 = response1.choices[0].message.content

        print(result1)

        # Store the results in the DataFrame
        current_df = pd.DataFrame({
            'kaggle_id': row['CurrentKernelVersionId'],
            'code': row['CleanCode'],
            'comp_id': row['CompetitionId'],
            'rating':row['PublicLeaderboardRank'],
            'data_type': result1,
        },index=[0])

        print(current_df)
        results_df = pd.concat([results_df, current_df], ignore_index=True)
        cnt+=1
        
    except APIError as e:
        print(f"Error: {e}. Skipping this code.")
        print(f"Problematic code: {code}")
        
    except Exception as e:
        print(e)

    # Save intermediate results every 5 steps to CSV
    if cnt % 5 == 0:
        results_df.to_csv(f'./data_type_res/data_type{cnt}.csv', index=False)
    

# Save the final results to a CSV file
results_df.to_csv('./data_type_res/data_type.csv', index=False)
