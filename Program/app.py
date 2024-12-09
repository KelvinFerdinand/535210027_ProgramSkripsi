from flask import Flask, render_template, send_from_directory, jsonify, request, session, redirect, url_for
import logging
import os
import pandas as pd
from python.train import read_data, train_model
import pickle
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'SkripsiKF2024'

# Load your Python model
model = pickle.load(open('models/gpa_model.pkl', 'rb'))

# Fixed value
check_outlier = True
seed = 42
y_loc = 0
C_val = [0.01, 0.1, 1, 10, 100]
gamma_val = [0.01, 0.05, 0.1, 0.5, 1]
epsilon_val = [0.01, 0.05, 0.1]

@app.route('/')
def index():
    return render_template('menu.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        
        print("Received data:", request.data)
        print("Parsed form data:", request.form)
        
        # Get the input data from the form
        algorithms = request.form.get('algorithms', default=0, type=float)
        computation1 = request.form.get('computation1', default=0, type=float)
        computation2 = request.form.get('computation2', default=0, type=float)
        datastructures = request.form.get('datastructures', default=0, type=float)

        # Make a prediction using the model
        prediction = model.predict([[algorithms, computation1, computation2, datastructures]])[0]

        # Return the result to the web page
        return jsonify({'gpa': prediction})
    return render_template('index.html')

@app.route('/train')
def train():
    return render_template('train_model.html')

@app.route('/save', methods=['POST'])
def save():
    # Check if the file is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Generate the timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Buat file path baru
    file_path = os.path.join("data", f"{timestamp}-{file.filename}")
    
    # Simpan file datanya
    file.save(file_path)
    
    # Baca file
    col_head = request.form.get('columnHeader') == 'on'
    print (col_head)

    # Ambil nilai proporsi data testing
    test_prop = float(request.form.get('test')) / 100.0

    # Ambil nilai jumlah variabel independen
    jumlah_x = int(request.form.get('totalVariables'))

    session['data'] = {
        'file_path': file_path,
        'col_head': col_head,
        'test_prop': test_prop,
        'jumlah_x': jumlah_x,
    }

    return jsonify({'success': True})

@app.route('/results')
def results():
    # Get results from the session
    data = session.get('data', None)

    if data is None:
        return jsonify({'error': 'No data found'}), 400

    file_path = data['file_path']
    col_head = data['col_head']
    test_prop = data['test_prop']
    jumlah_x = data['jumlah_x']

    # Baca file
    df = read_data(file_path, col_head)

    x_loc = list(range(1, jumlah_x + 1))

    results = train_model(df, check_outlier, test_prop, seed, y_loc, x_loc, C_val, gamma_val, epsilon_val)

    context = {
        'raw': results['raw_data'],
        'missing': results['missing_data'],
        'outlier': results['outlier_data'],
        'score_outlier': results['score_outlier'],
        'clean': results['clean_data'],
        'train': results['train_data'],
        'gs': results['grid_search'],
        'sv': results['support_vectors'],
        'test': results['test_results'],
        'b': results['b'],
        'x_cont': results['x_cont'],
        'x_corr': results['x_corr'],
        'SSR': results['SSR'],
        'SSE': results['SSE'],
        'SST': results['SST'],
        'dfR': results['dfR'],
        'dfE': results['dfE'],
        'MSR': results['MSR'],
        'MSE': results['MSE'],
        'F': results['F'],
        'MAE': results['MAE'],
        'MAPE': results['MAPE'],
        'e_MSE': results['e_MSE'],
        'R2': results['R2'],
        'kernel': results['kernel'],
        'C': results['C'],
        'gamma': results['gamma'],
        'epsilon': results['epsilon']
    }

    return render_template('result.html', **context)

@app.route('/get-plot/<path:filename>')
def get_plot(filename):
    # Mengirim gambar dari folder parent
    plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'python/img/'))
    return send_from_directory(plot_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)