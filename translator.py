from transformers import T5Tokenizer, T5ForConditionalGeneration
from config import MODEL_CONFIG
from database import execute_and_fetch_query

tokenizer = T5Tokenizer.from_pretrained(MODEL_CONFIG['model_name'])
model = T5ForConditionalGeneration.from_pretrained(MODEL_CONFIG['model_name'])


def generate_sql_statements(nl_input: str) -> list:
    """ Generate the most probable SQL statement and two diverse statements 
    from the input natural language query."""
    encoding = tokenizer.encode_plus(
        nl_input, padding='max_length', max_length=MODEL_CONFIG['max_length'], return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

    sql_statements = []

    # Generate the most probable output
    most_probable_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=MODEL_CONFIG['max_length'],
        do_sample=False,
        early_stopping=True,
        num_return_sequences=1
    )
    sql = tokenizer.decode(
        most_probable_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    sql_statements.append(sql)

    # Generate two more diverse outputs
    diverse_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=MODEL_CONFIG['max_length'],
        do_sample=True,
        top_k=MODEL_CONFIG['top_k'],
        top_p=MODEL_CONFIG['top_p'],
        early_stopping=True,
        num_return_sequences=MODEL_CONFIG['number_of_outputs']-1
    )
    for output in diverse_outputs:
        sql = tokenizer.decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        sql_statements.append(sql)

    return sql_statements


def process_nl_query(nl_query: str):
    """
    Translates Natural Language query to SQL, executes it and fetches the results.
    """

    sql_statements = generate_sql_statements(nl_query)
    results = [execute_and_fetch_query(statement)
               for statement in sql_statements]

    return sql_statements, results
