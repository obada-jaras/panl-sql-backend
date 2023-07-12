import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from translator import process_nl_query

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    API endpoint to process Natural Language Query
    """
    data = request.get_json()
    nl_query = data.get("query")
    logging.info(f"Received NL query: {nl_query}")

    sql_statements, results = process_nl_query(nl_query)

    return jsonify([{"sql": s, "result": r} for s, r in zip(sql_statements, results)])


if __name__ == "__main__":
    app.run(debug=True)
