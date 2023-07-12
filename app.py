from flask import Flask, request, jsonify
from translator import process_nl_query

app = Flask(__name__)


@app.route('/api/query', methods=['GET'])
def api_query():
    """
    API endpoint to process Natural Language Query
    """
    data = request.get_json()
    nl_query = data.get('nl_query')

    sql_statements, results = process_nl_query(nl_query)

    return jsonify(
        sql_outputs=[{'sql': s, 'result': r}
                     for s, r in zip(sql_statements, results)]
    )


if __name__ == '__main__':
    app.run(debug=True)
