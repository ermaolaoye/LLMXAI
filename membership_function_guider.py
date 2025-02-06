# %% Prompt
MEMBERSHIP_FUNCTION_GUIDER_PROMPT = """
You are a fuzzy system membership function extraction expert. Your task is to analyze a JSON input that represents a tabular database and generate guidance on how to extract fuzzy membership functions for each column (variable) using the Fuzzy C-Means (FCM) algorithm.
The JSON input has the following structure:

```json
{
 "database_name": "Some descriptive name or label",
 "database_description": "Some additional context or domain-specific details",
 "database_data": {
         "columns": ["Column1", "Column2", ..., "TargetVariable"],
         "rows": [
             {"Column1": "value11", "Column2": "value12", ..., "TargetVariable": "value1N"},
             {"Column1": "value21", "Column2": "value22", ..., "TargetVariable": "value2N"},
             ...
         ]
     },
 "target_value": "Name or index of the target column"
}
```

Key Points:

database_name: Provides a descriptive name for the dataset that can be used for naming conventions.
database_description: Contains background or domain-specific information that may influence the design (e.g., naming or shape selection) of the membership functions.
database_data: Holds the actual dataset. The columns array lists all variables, including input variables and a target (prediction) variable. The rows array contains the dataset’s values.
target_value: Indicates the target (prediction) column. This variable should primarily serve as context (for instance, guiding naming) and may be excluded from the membership function extraction if appropriate.

Your job is to decide for each non-target column:

Number of clusters (membership functions) for fuzzy extraction using FCM.
Membership function shapes (e.g., Gaussian, triangular, trapezoidal).
Appropriate names for each cluster, informed by the dataset’s domain and distribution.
Special handling of binary (one-hot-encoded) columns:
- If a column contains only two unique values (e.g., True and False, 0 and 1), treat it as a crisp membership function rather than fuzzy.
- Assign two crisp membership functions: one for each possible value.
The output must be a valid JSON that contains your recommendations. Structure your output similar to the following template:

```json
{
 "database_name": "<same as input>",
 "membership_functions": {
   "Column1": {
     "num_clusters": <number>,
     "clusters": [
       {"name": "<cluster name>", "shape": "<membership function shape>"},
       {"name": "<cluster name>", "shape": "<membership function shape>"},
       ...
     ]
   },
   "BinaryColumn": {
     "num_clusters": 2,
     "clusters": [
       {"name": "False", "shape": "crisp"},
       {"name": "True", "shape": "crisp"}
     ]
   },
   "Column2": {
     "num_clusters": <number>,
     "clusters": [
       {"name": "<cluster name>", "shape": "<membership function shape>"},
       ...
     ]
   }
 }
}
```

Additional Instructions:

Use the database_description and database_name to inform the naming and selection of membership function shapes.
Exclude the target column (as specified by target_value) from membership function creation unless it is contextually useful for guiding the process.
Ensure binary columns (with only two values like True/False or 0/1) are represented as crisp membership functions rather than fuzzy clusters.
Ensure all the columns are included in your output.
Ensure that the final output is valid JSON.

Generate the membership function extraction guide based on these instructions.
"""

# %% Setting up the OpenAI API for LM Studio
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:12366/v1",
    api_key="lm-studio"
)
# %% Initializing AI Server

from llm_server import LLMServer

llm = LLMServer(token="sk-763f51916aaa4a548cbb4bf365fb9e81")

llm.set_system(MEMBERSHIP_FUNCTION_GUIDER_PROMPT)

# %% Processing Database
import json
import csv
import random

def serialize_csv_to_json(file_path, database_name, database_description, target_variable, max_rows=100):
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        if target_variable not in columns:
            raise ValueError("Target variable not found in the CSV file.")

        data = [row for row in reader]

        if len(data) > max_rows:
            data = random.sample(data, max_rows)

    serialized_json = {
        "database_name": database_name,
        "database_description": database_description,
        "database_data": {
            "columns": columns,
            "rows": data
        },
        "target_value": target_variable
    }

    return json.dumps(serialized_json, indent=4)

file_path = "databases/bank_marketing_preprocessed.csv"
database_name = "Bank Marketing Dataset"
database_description = "This dataset is related to direct marketing campaigns of a Portuguese banking institution."
target_variable = "y"

serialized_json = serialize_csv_to_json(file_path, database_name, database_description, target_variable, max_rows=5)

# Save the serialized JSON to a new file
with open("serialized_database.json", "w") as file:
    file.write(serialized_json)

# %% Generate Completion in LM Studio
completion = client.chat.completions.create(
    model="deepseek-r1-distill-qwen-32b",
    # model="deepseek-r1-distill-qwen-7b",
    messages=[
        {"role": "system", "content": MEMBERSHIP_FUNCTION_GUIDER_PROMPT},
        {"role": "user", "content": serialized_json}
    ],
    temperature=0.3,
)

# %% Generating Response from AI Server
response = llm.ask(serialized_json)

with open("membership_function_guidance_response_kcl.md", "w", encoding='utf-8') as file:
    file.write(response)

# %% Extracting the JSON response
result = completion.choices[0].message.content + "\n```"
with open("membership_function_guidance_response_lm.md", "w", encoding='utf-8') as file:
    file.write(result)
# %% Use 're' module to only return the JSON part
import re

pattern = r"```json\s*([\s\S]+?)\s*```"

match = re.search(pattern, result)

if match:
    # Save the extracted JSON to a new file
    json_text = match.group(1)
    with open("membership_functions.json", "w") as file:
        file.write(json_text)
else:
    raise ValueError("JSON not found in the response.")

# %% Verify the extracted JSON
import json
import csv

with open("membership_functions_config.json", "r") as file:
    data = json.load(file)
    json_columns = list(data['membership_functions'].keys())


with open(file_path, "r") as file:
    reader = csv.DictReader(file)
    database_columns = reader.fieldnames

    # Exclude the target column
    database_columns.remove(target_variable)

print("JSON Columns:", json_columns)
print("Database Columns:", database_columns)

if json_columns != database_columns:
    raise ValueError("Columns mismatch between input and output JSON.")
