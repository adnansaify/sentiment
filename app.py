from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle

linear_svm=pickle.load(open('linear_svm.pkl','rb'))
vectorizer=pickle.load(open('vectorizer.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    if request.method=='POST':
        message=request.form['review_text']
        text=[message]
        print(text)
        test_features=vectorizer.transform(text)
        df_text=pd.DataFrame(test_features.toarray(),columns=vectorizer.get_feature_names())
        my_prediction=linear_svm.predict(df_text)
        print(my_prediction)
        if my_prediction==0:
            prediction = 'NEGATIVE'
        elif my_prediction == 1:
            prediction = 'POSITIVE'
        print(prediction)
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)



