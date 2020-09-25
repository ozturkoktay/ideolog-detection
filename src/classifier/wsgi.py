from main import app

if __name__ == '__main__':
    #  env FLASK_APP=webservice.py flask run
    app.run(threaded=True)
