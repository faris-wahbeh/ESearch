import os
import flask
import openai
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from xml.etree import ElementTree as ET
from flask_cors import CORS
from flask import request, jsonify


app = flask.Flask(__name__)


ALLOWED_ORIGINS = ["http://localhost:3000",
                   "https://git.heroku.com/econsearchapp.git"]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})


# replace with your OpenAI API key
openai.api_key = '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'

# Schema for indexing
schema = Schema(year=TEXT(stored=True), month=TEXT(stored=True),
                page=ID(stored=True), para_num=ID(stored=True),
                text=TEXT(stored=True))

index_dir = "indexdir"

# Create or open the index directory
if not os.path.exists(index_dir):
    os.mkdir(index_dir)
# This line should be outside the if-else block
ix = create_in(index_dir, schema)


def index_xml_files(base_dir="texts"):
    writer = ix.writer()
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if os.path.isdir(year_path):
            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if os.path.isdir(month_path):
                    for xml_file in os.listdir(month_path):
                        xml_path = os.path.join(month_path, xml_file)
                        # Print the path of the XML file
                        print(f"Processing: {xml_path}")
                        try:
                            tree = ET.parse(xml_path)
                            root = tree.getroot()
                            for para in root.findall('paragraph'):
                                writer.add_document(year=year, month=month,
                                                    page=xml_file.split('.')[
                                                        0],
                                                    para_num=para.attrib['number'],
                                                    text=para.text.lower())
                        except ET.ParseError as e:
                            # Print the error message and surrounding XML content
                            with open(xml_path, 'r') as f:
                                lines = f.readlines()
                            error_line = e.position[0]
                            start = max(0, error_line - 3)
                            end = min(len(lines), error_line + 3)
                            print(f"XML ParseError at {e.position}:")
                            for i in range(start, end):
                                print(f"{i + 1}: {lines[i].strip()}")
                            return
    writer.commit()


index_xml_files()


def search_paragraphs(keywords, year=None, month=None):
    keywords_list = [word.lower() for word in keywords.split()]
    searcher = ix.searcher()
    query_parts = []
    for keyword in keywords_list:
        query_parts.append(f"text:{keyword.lower()}")
    if year:
        query_parts.append(f"year:{year}")
    if month:
        query_parts.append(f"month:{month}")
    query_str = " AND ".join(query_parts)
    parser = QueryParser("text", ix.schema)
    query = parser.parse(query_str)
    results = searcher.search(query, limit=None)
    paragraphs = [(r['year'], r['month'], r['page'],
                   r['para_num'], r['text']) for r in results]
    tokenized_corpus = [word_tokenize(para[4]) for para in paragraphs]

    # Check if tokenized_corpus is empty
    if not tokenized_corpus:
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(keywords_list)
    ranked_paragraphs = [para for _, para in sorted(
        zip(scores, paragraphs), key=lambda pair: pair[0], reverse=True)]
    return list(set(ranked_paragraphs))


def chat_with_gpt(question, selected_paragraphs):
    system_message_text = "\n".join(selected_paragraphs)
    system_message = {"role": "system", "content": system_message_text}

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                system_message,
                {"role": "user", "content": question},
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print("Error interacting with GPT:", e)
        return "Sorry, I couldn't provide an answer at the moment."


@app.route('/search', methods=['POST'])
def search_route():
    data = request.json
    keywords = data.get('keywords', '')
    year = data.get('year')
    month = data.get('month')
    results = search_paragraphs(keywords, year, month)
    print(results)  # Add this line to debug
    return jsonify(results)


@app.route('/chatgpt', methods=['POST'])
def chatgpt_route():
    try:
        data = request.json
        question = data.get('question', '')
        selected_paragraphs = data.get('selected_paragraphs', [])
        answer = chat_with_gpt(question, selected_paragraphs)
        return jsonify({"answer": answer})
    except Exception as e:
        print("Error in chatgpt_route:", e)
        return jsonify({"error": "Error processing request."}), 500


@app.route('/')
def index():
    return flask.send_from_directory('econsearch-frontend', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return flask.send_from_directory('econsearch-frontend', path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)

