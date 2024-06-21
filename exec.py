import json
import openai
from langchain_community.utilities import SQLDatabase

# Set your OpenAI API key
openai.api_key = 'sk-proj-5HpTAByJImynOhTp9CpJT3BlbkFJgWQndJVGOYYCQel1BXun'



def load_test_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        input_queries = [item['input_query'] for item in data]
        true_sql_queries = [item['true_sql'] for item in data]
        return input_queries, true_sql_queries
    except Exception as e:
        print(f"Error loading test data: {e}")
        return [], []

def generate_sql_from_model(input_query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts natural language queries into SQL queries."},
                {"role": "user", "content": f"Convert the following natural language query to SQL: '{input_query}'"}
            ],
            max_tokens=150,
            temperature=0.2,
        )
        sql_query = response['choices'][0]['message']['content'].strip()
        return sql_query
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return ""

def execute_sql(query, db):
    try:
        result = db.query(query)
        return result.fetchall()
    except Exception as e:
        print(f"Error executing query: {query}\n{e}")
        return None

# Database connection
try:
    db = SQLDatabase.from_uri(
        'postgresql+psycopg2://postgres:postgres@localhost:5432/Leanios_development?options=-csearch_path=dummy'
    )
except Exception as e:
    print(f"Error connecting to database: {e}")

# Load test data
input_queries, true_sql = load_test_data('test_data.json')

# Generate predictions
predicted_sql = [generate_sql_from_model(query) for query in input_queries]

# Calculate Exact Match Accuracy
exact_match_accuracy = sum(1 for true, pred in zip(true_sql, predicted_sql) if true == pred) / len(true_sql)

# Calculate Execution Accuracy
execution_match_count = 0
total_queries = len(true_sql)

for true_query, pred_query in zip(true_sql, predicted_sql):
    true_result = execute_sql(true_query, db)
    pred_result = execute_sql(pred_query, db)
    if true_result == pred_result:
        execution_match_count += 1

execution_accuracy = execution_match_count / total_queries

print(f'Exact Match Accuracy: {exact_match_accuracy}')
print(f'Execution Accuracy: {execution_accuracy}')
