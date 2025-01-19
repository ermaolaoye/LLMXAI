# %% Prompt
MEMBERSHIP_FUNCTION_EXTRACTOR_PROMPT = """
You are a type-1 fuzzy system membership function extractor. The user will provide data in a JSON format with the following structure:

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

Where:
- **database_name**: A descriptive name for the dataset; can guide variable naming or membership function interpretation.
- **database_description**: Additional background or domain information for extracting context-specific membership functions.
- **database_data**: The dataset where columns lists all variables(input and target variables). Rows contain the actual data for each variable in the columns array.
- **target_value**: Indicates which column in the database is the target (prediction) variable. This column should be used primarily as reference (e.g., to guide labeling or context). You may exclude the target variable from membership function creation if appropriate.

Your task is to:
1. **Read** the JSON object from the user input.
2. **Use** the "database_name" and "database_description" for context, as needed.
3. **Parse** the data from the "database_data" to understand the variables and their values.
4. **Identify** which column is the target (from "target_value").
5. **Generate** type-1 fuzzy membership functions **only** for the non-target columns (or handle the target column differently, per your approach).
6. **Output** the fuzzy system definition as **raw JSON** with **no extra text**, in the structure:

```json
{
  "variables": [
    {
      "name": "VariableName",
      "range": [minValue, maxValue],
      "membership_functions": [
        {
          "mf_name": "NameOfMembershipFunction",
          "mf_type": "TypeOfMF",
          "parameters": [param1, param2, ...]
        },
        ...
      ]
    },
    ...
  ]
}
```

Where:
- **variables** is an array describing each variable (column or conceptual grouping).
- **name** is the variable’s name or a label (you may consider the database context to create meaningful labels).
- **range** is [minValue, maxValue] for that variable’s domain, based on the CSV data.
- **membership_functions** is an array of objects, each describing one membership function.
  - **mf_name** can be something like “Low”, “Medium”, “High” or any label (possibly guided by the database context).
  - **mf_type** could be "triangular", "trapezoid", "gaussian", etc.
  - **parameters** is an array of numeric values for the chosen membership function type.

You must **only** return valid JSON in the specified structure with **no additional text** or explanations.
"""

# %% Initializing AI Server
from llm_server import LLMServer

llm = LLMServer(token="sk-763f51916aaa4a548cbb4bf365fb9e81")

llm.set_system(MEMBERSHIP_FUNCTION_EXTRACTOR_PROMPT)
# %% Processing Database
import json
import csv
import random

def serialize_csv_to_json(file_path, database_name, database_description, target_variable, max_rows=100):
    with open(file_path, "r") as file:
        reader = csv.DictReader(file, delimiter=";")
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

file_path = "databases/bank-full.csv"
database_name = "Bank Marketing Dataset"
database_description = "This dataset is related to direct marketing campaigns of a Portuguese banking institution."
target_variable = "y"

serialized_json = serialize_csv_to_json(file_path, database_name, database_description, target_variable, max_rows=10)

# Save the serialized JSON to a new file
with open("serialized_database.json", "w") as file:
    file.write(serialized_json)

# %% Querying AI Server
response = llm.ask(serialized_json)

print(response)

# Save the response to a JSON file
with open("membership_functions.json", "w") as file:
    file.write(response)
