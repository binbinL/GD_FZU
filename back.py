from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, session, blueprints

from apps.user import bp as user_bp

app = Flask(__name__)
app.config['JSON＿AS＿ASCII'] = False
app.register_blueprint(user_bp)


@app.route('/', methods=['GET'])
def to_login():
    return redirect('http://127.0.0.1:5000/user/login')


if __name__ == '__main__':
    app.config["SECRET_KEY"] = 'Hello World'
    app.run(debug=True, port=5000)
