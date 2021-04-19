import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
from datetime import datetime

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

 
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints,intents_json,msg):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
 
        dateIn = msg
        dateControl = isinstance(dateIn, type(datetime.date))
        print(dateIn)
        print(dateControl)
        if dateControl == True:
            i[tag] = 'dateIn'
            if(i[tag] == 'dateIn'):
                result = "You will arrive on " + dateIn + random.choice(i['responses'])
                break

        day_num = msg # Klavyeden sayı girişi alındığında gün olarak alıyor. Group reservation vb. için özelleştirilebilir.
        if(day_num.isnumeric()==True):
            i[tag] = 'days'
            if(i['tag'] == 'days'):
                result = "Ok "+ day_num + " days. " + random.choice(i['responses'])
                break   

        elif(i['tag']== tag):
            result = random.choice(i['responses'])
            break 
 
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents,msg)

    #Kullanıcının yazdıkları ve verilen cevaplrın kaydedilmesi için 
    with open("records.json",'r') as fp:
        chatBotRecords= json.load(fp)

    chatBotRecords["records"].append({
        "message":msg,
        "response":res
    })
    with open("records.json",'w') as fp:
        json.dump(chatBotRecords,fp,indent=2)

    return res


from flask import Flask, render_template, request

app = Flask(__name__)

#İlk çalıştırıldığında ana sayfa olacak şekilde düzenlendi.
@app.route('/')
def index():
    return render_template('pop.html')

@app.route('/index')#deneme-test sayfasıdır kaldırılabilir.
def pop():
    return render_template('index.html')
 
@app.route('/get')#Js ile ileşim fonksiyonu
def get_bot_respose():
    userText = request.args.get('msg')
    msg = userText

    res = chatbot_response(msg)
    botText = res

    return str(botText)



 #index.html işlemi için , kaldırılabilir.
@app.route('/send',methods=['POST','GET'])
def send():
    
    msg = request.form['msg'] 
    res = chatbot_response(msg)

    if request.method == 'POST':
        msg = request.form['msg']
        request.method == 'GET'

    if request.method =='GET':
        res = chatbot_response(msg)
        res = str(res) 

    return render_template('index.html', msg=msg, res=res ) 

if __name__ == '__main__':
    app.run(debug=True, port=5004)



