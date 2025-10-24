from flask import Flask, request, jsonify
from flask_cors import CORS
from classifier import prediction_by_gemini
import logging
import os
import traceback

# Basic logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s - %(message)s'
)
logger = logging.getLogger('backend_server')

app = Flask(__name__)
# Default to allowing CORS from localhost:4200 (frontend) but can be overridden by env variable
frontend_origin = os.environ.get('FRONTEND_ORIGIN', 'http://localhost:4200')
CORS(app, origins=[frontend_origin], allow_headers=['Content-Type'], methods=['GET', 'POST', 'OPTIONS'])

@app.route('/classify', methods=['POST'])
def classify_conversation():
    try:
        data = request.get_json(force=True, silent=True)
        logger.debug('Raw request JSON: %s', data)

        if data is None:
            return jsonify({'error': 'JSON non valido o mancante'}), 400

        messages = data.get('messages', [])

        # Format the conversation with speaker labels
        conversation = ''
        for i, message in enumerate(messages):
            speaker = message.get('speaker', f'Speaker{i+1}')
            content = message.get('content', '')
            if isinstance(content, str) and content.strip():
                conversation += f"{speaker}: \"{content}\"\n"

        if not conversation.strip():
            logger.info('Empty conversation after parsing messages')
            return jsonify({'error': 'Nessuna conversazione fornita'}), 400

        logger.debug('Formatted conversation:\n%s', conversation)

        # Call the Gemini classifier
        label, explanation = prediction_by_gemini(conversation)

        logger.debug('Classifier returned label=%s, explanation=%s', label, explanation)

        if label is None:
            logger.error('Classifier returned None label')
            return jsonify({'error': 'Errore durante la classificazione'}), 500

        return jsonify({
            'explanation': explanation,
            'is_toxic': bool(label == 1)
        })

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception('Unhandled exception in /classify: %s', e)
        return jsonify({'error': f'Errore del server: {str(e)}', 'traceback': tb}), 500


@app.route('/classify-image', methods=['POST'])
def classify_image():
    try:
        data = request.get_json(force=True, silent=True)

        if data is None:
            return jsonify({'error': 'JSON non valido o mancante'}), 400

        image_data = data.get('image')
        if not image_data:
            logger.info('No image provided in request')
            return jsonify({'error': 'Nessuna immagine fornita'}), 400

        label, explanation = prediction_by_gemini(image_data, is_image=True)
        return jsonify({
            'explanation': explanation,
            'is_toxic': bool(label == 1)
        })

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception('Unhandled exception in /classify-image: %s', e)
        return jsonify({'error': f'Errore del server: {str(e)}', 'traceback': tb}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    logger.info('Avvio server backend sulla porta %s...', port)
    app.run(debug=True, port=port, host='0.0.0.0')