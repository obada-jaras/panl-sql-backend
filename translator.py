import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from config import MODEL_CONFIG
from database import execute_and_fetch_query
import logging

tokenizer = T5Tokenizer.from_pretrained(MODEL_CONFIG['model_name'])
model = T5ForConditionalGeneration.from_pretrained(MODEL_CONFIG['model_name'])


def generate_sql_statements(nl_input: str, num_outputs: int) -> list:
    """ Generate multiple SQL statements from the input natural language query."""
    encoding = tokenizer.encode_plus(
        nl_input, padding='max_length', max_length=512, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

    sql_statements = []
    for id in range(num_outputs):
        torch.manual_seed(id+1)
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=512,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=1
        )
        sql = tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        sql_statements.append(sql)

    return sql_statements


def process_nl_query(nl_query: str):
    """
    Translates Natural Language query to SQL, executes it and fetches the results.
    """
    logging.info(f"Received NL query: {nl_query}")

    sql_statements = generate_sql_statements(
        nl_query, MODEL_CONFIG['number_of_outputs'])
    results = [execute_and_fetch_query(statement)
               for statement in sql_statements]

    return sql_statements, results
