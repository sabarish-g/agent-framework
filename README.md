### Use Poetry to set up environment.

## Agentic Framework using Autogen 0.4

## Requierd: data folder with a csv file for demoing SQL and pandas agent capabilities
### Step1: Use csv_to_sql.py to convert the csv file into a sqlite db
### Step2: Modify prompt to handle the schema of the table you are using in sqlagent.py before running it.
### Step3: Same things can be achieved using pandas by using duckdb
### Step4: Set up a faiss vector index in data folder and use ragagent.py to run rag using autogen
### Step5: Run main.py which selectively chooses different agents (sql vs rag) depending on the question.