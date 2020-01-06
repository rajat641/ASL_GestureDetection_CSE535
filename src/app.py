from flask import Flask
import main
import test_main
from flask import request
import os
import shutil

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def run_app():
    # json_data = main.main()
    # print(json_data)
    return "/pred/ for Predictions '\n' /train/ for Training"

@app.route('/train/', methods=['GET', 'POST'])
def run_app_train():
    output = main.main()
    print(output)
    return output


@app.route('/pred1/', methods=['GET','POST'])
def pred_app_o():
    ## Create temporary Directory to store file
    uploads_dir = os.path.join(app.instance_path, 'uploads')
    os.makedirs(uploads_dir)
    if request.method == 'POST':
        if request.files.get('file') == None:
            dic = {}
            dic['Error']="No json file with name 'file' provided"
            return dic
        ## Save the file to local directory
        json_file = request.files.get('file')
        json_name = json_file.filename
        mimetype = json_file.content_type
        # json_file.save(os.path.join(app.config['UPLOAD_FOLDER'],json_name))
        json_file.save(os.path.join(uploads_dir,json_name))
    file_path = uploads_dir+'/'+json_name
    # print(file_path)
    ## Get the predictions

    json_data = test_main.pred(file_path)
    print(json_data)
    shutil.rmtree(app.instance_path)
    return json_data


@app.route('/pred/', methods=['GET','POST'])
def pred_app():
    if request.method == 'POST':
        data = request.get_json()
    json_data = test_main.pred(data)
    print(json_data)
    # shutil.rmtree(app.instance_path)
    return json_data


if __name__ == '__main__':
    app.run()
