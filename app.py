import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify, render_template

# Step 1: Prepare and train the model
data = {
    'url': [
        'https://www.google.com', 
        'http://malicious.com/badsite', 
        'https://facebook.com/login', 
        'http://phishing.com/login.php',
        'https://github.com', 
        'http://badwebsite.com/malware',
        'https://linkedin.com/in/login', 
        'http://fakebank.com/login',
        'https://twitter.com', 
        'http://harmfulsite.org/download'
    ],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = safe, 1 = unsafe
}

# Create a DataFrame
df = pd.DataFrame(data)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(token_pattern=r'[A-Za-z0-9]+', max_features=3000)
X = vectorizer.fit_transform(df['url'])
y = df['label']

# Train a simple logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model trained with accuracy: {accuracy * 100:.2f}%')

# Initialize Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle URL prediction
@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    test_features = vectorizer.transform([url])
    prediction = model.predict(test_features)
    result = "unsafe" if prediction[0] == 1 else "safe"
    return jsonify({'url': url, 'result': result})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
