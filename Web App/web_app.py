from flask import Flask
from flask import request
from flask import render_template,request
from tensorflow.keras import datasets, layers, models
import librosa
import audioread
import numpy as np
import os 
from os.path import splitext
import audio_metadata
from tinytag import TinyTag
import webbrowser, random,threading
from skimage.transform import resize

app=Flask(__name__)

@app.route("/",methods=['GET', 'POST'])
def home():
 """Home page of app with form"""
 # Create form
 if request.method == "GET":
  return render_template('index.html')
 elif request.files:
  p_bpm="not requested"
  p_keys="not requested"
  file=request.files["file"]
  filename = file.filename
  file.save(file.filename)
  model_bpm = models.load_model("model_bpm.h5") 
  model_key = models.load_model("model_key.h5")
  y,sr=librosa.load(filename,sr=22050)
  ps = librosa.feature.melspectrogram(y=y,n_fft=1024, hop_length=1024//2,sr=sr,power=1,n_mels=40, fmin=20, fmax=5000)
  mels = np.log(ps + 1e-9)
  mels = librosa.power_to_db(ps, ref=np.max)
  mels_resized= np.transpose(resize(mels, (mels.shape[0], int(mels.shape[1] * 0.39629)),anti_aliasing=True))[:256,:]
  mels_resized=np.reshape(mels_resized, (1,mels_resized.shape[0],mels_resized.shape[1]))
  if request.form.getlist('BPM'):
   result=model_bpm.predict(mels_resized)
   p_bpm=np.argmax(result, axis=-1)[0]+60
  if request.form.getlist('KEY'):
   result=model_key.predict(mels_resized)
   p_keys=np.argmax(result, axis=-1)[0]
   lst_name_chords=['A major','Bb major','B major','C major','Db major','D major','Eb major','E major','F major','Gb major','G major','Ab major','A minor','Bb minor','B minor','C minor','Db minor',
   'D minor','Eb minor','E minor','F minor','Gb minor','G minor','Ab minor']
   p_keys=lst_name_chords[p_keys]
   #get metadata
  tag=TinyTag.get(filename)

  if not tag.title:
   title=splitext(filename)[0]
  else:
   title=tag.title
  if "mp3" in filename:
   image= "{{ url_for('static',filename='static/styles/Images/mp3_icon.png') }}"
  else:
   image="{{ url_for('static',filename='static/styles/Images/wav_icon.png') }}"
  return render_template('result.html', file=title,image=image, prediction_bpm=p_bpm, prediction_key= p_keys)
  


 # Send template information to index.html
 return render_template('index.html')
if __name__=="__main__":

 port = 5000 + random.randint(0, 999)
 url = "http://127.0.0.1:{0}".format(port)

 threading.Timer(1.25, lambda: webbrowser.open(url) ).start()

 app.run(port=port, debug=False)