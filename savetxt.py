@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No video uploaded", 400

        file = request.files['video']
        if file.filename == '':
            return "No selected file", 400

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(save_path)

        predicted_text = predict_from_video(save_path)

        # Save the predicted text to a .txt file
        txt_filename = os.path.splitext(file.filename)[0] + "_subtitle.txt"
        txt_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)

        with open(txt_path, "w") as f:
            f.write(predicted_text)

        # Return template with subtitle file name
        return render_template('upload.html',
                               filename=file.filename,
                               subtitle=predicted_text,
                               subtitle_file=txt_filename)

    return render_template('upload.html')
