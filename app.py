from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ML import *  # training_ML(): return model
from Google_Vison_API import *  # main(path): return input_data

'''Training model'''
model = training_ML()
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        # 입력받은 이미지 파일이 저장된 경로로 
        f = request.files['image']
        f.save('C:/Users/PC/Desktop/visionapi/ImageClassificationML/static/image/' +  f.filename)
        
        ''' image upload & perdict '''
        input_data = image_info(f'static/image/{f.filename}')  # 가현이 컴으로 실행!  

        print(f"입력받은 이미지: {f.filename}, input_data: {input_data}")

        Y_pred = model.predict(input_data)
        if max(Y_pred[0]) < 0.6:
            Y_pred[0][5] = 1
        predicted = Y_pred.argmax(axis=-1)
        print(f"예측: {predicted}")

        src = "image/" + f.filename
        val_list = [predicted, src]
        return render_template('after_upload.html', value=val_list)

    else:
        return render_template('file_upload.html')
    
        
if __name__ == '__main__':
    app.run(port='5000', debug=False)