from flask import Flask, request, jsonify
from next_word_prediction import GPT2
from current_word_prediction.model import get_model, get_current_word
app = Flask(__name__)

gpt2 = GPT2()
current_word_model = get_model()

@app.route('/next_word', methods=['POST'])
def next_word_receiver():
    message = request.form.get('text')
    words = gpt2.predict_next(message, 2)
    print(words)
    return jsonify({'words': "-".join(words)})


@app.route('/current_word', methods=['POST'])
def current_word_receiver():
    message = request.form.get('text')
    if " " in message:
        message = message.split(" ")[-1]
    print(message)
    word = get_current_word(message, current_word_model)
    
    word1 = " "
    if len(message) > 1:
        message = message[:-1]
        word1 = get_current_word(message, current_word_model)
 
    return jsonify({'words': "-".join([word, word1])})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)