import os
import io
import PyPDF2
from pydantic import BaseSettings
import time
import torch
from pydub import AudioSegment
import torchaudio
from fastapi import FastAPI,File, UploadFile
from fastapi.responses import StreamingResponse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import spacy
import joblib
config = XttsConfig()
config.load_json("config.json")
model = Xtts.init_from_config(config)

pipe_lr = joblib.load(open("./emotion_classifier_pipe_lr.pkl","rb"))
def read_wav_file():
    # For demonstration, let's assume you are reading the binary WAV from a file
    with open("/Users/manasagarwal/Documents/xtts/true.wav", "rb") as wav_file:
        wav_content = wav_file.read()
    return wav_content
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]
def split_sentences_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences
def create(index,txt,emotion,gpt_cond_latent,speaker_embedding):
    try:
        if(emotion=='happy'):
            
            print("Inference...")
            t0 = time.time()
            chunks = model.inference_stream(
            txt,
            "en",
            gpt_cond_latent,
            speaker_embedding
            )

            wav_chuncks = []
            for i, chunk in enumerate(chunks):
                if(chunk is None):
                    continue
                
                if i == 0:
                    print(f"Time to first chunck: {time.time() - t0}")
                print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
                wav_chuncks.append(chunk)
            wav = torch.cat(wav_chuncks, dim=0)
            torchaudio.save(f"xtts_streaming{index}.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
        elif(emotion=='angry'):

            print("Inference...")
            t0 = time.time()
            chunks = model.inference_stream(
            txt,
            "en",
            gpt_cond_latent,
            speaker_embedding
            )

            wav_chuncks = []
            for i, chunk in enumerate(chunks):
                if(chunk is None):
                    continue
                if i == 0:
                    print(f"Time to first chunck: {time.time() - t0}")
                print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
                wav_chuncks.append(chunk)
            wav = torch.cat(wav_chuncks, dim=0)
            torchaudio.save(f"xtts_streaming{index}.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
        elif(emotion=='sad'):
            
            print("Inference...")
            t0 = time.time()
            chunks = model.inference_stream(
            txt,
            "en",
            gpt_cond_latent,
            speaker_embedding
            )


            wav_chuncks = []
            for i, chunk in enumerate(chunks):
                if(chunk is None):
                    continue
                if i == 0:
                    print(f"Time to first chunck: {time.time() - t0}")
                print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
                wav_chuncks.append(chunk)
            wav = torch.cat(wav_chuncks, dim=0)
            torchaudio.save(f"xtts_streaming{index}.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
        else:
            
            print("Inference...")
            t0 = time.time()
            chunks = model.inference_stream(
            txt,
            "en",
    gpt_cond_latent,
    speaker_embedding
)
            print(type(chunks))
            wav_chuncks = []
            # for i in chunks:
                # print(i)
            for i, chunk in enumerate(chunks):
                if(chunk is None):
                    continue
                print(chunks,15)
                if i == 0:
                    print(f"Time to first chunck: {time.time() - t0}")
                print(chunks,15)
                print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
                wav_chuncks.append(chunk)
            wav = torch.cat(wav_chuncks, dim=0)
            torchaudio.save(f"xtts_streaming{index}.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
        print(index)
    except Exception as e:
        raise TypeError(emotion,txt,e)
app = FastAPI()
mp3_file_path="/Users/manasagarwal/Documents/fastapi/f3 copy.wav"
async def tex_to_wav(tex):
    sizedlist=['']
    x=tex
    x=x.replace('\n',' ')
    x=split_sentences_spacy(x)
    sizedlist=[' ']
    for i in x:
        if len(sizedlist[-1]+i)<100:
            sizedlist[-1]+=i
        else:
            sizedlist.append(i)    
    anger_list=[]
    sad_list=[]
    happy_list=[]
    neutral_list=[]
    final_list=[]
    if(sizedlist[0]==' '):
        sizedlist.remove(' ')
    for j,i in enumerate(sizedlist):
        prediction = predict_emotions(i)
        if('angry'==prediction):
            anger_list.append(j)
        elif prediction == 'disgust' or prediction == 'sad' or prediction == 'sadness' or prediction == 'shame':
            sad_list.append(j)
        elif prediction == 'happy' or prediction == 'joy' or prediction=='surprise':
            happy_list.append(j)
        else:
            neutral_list.append(j)
    l=len(sizedlist)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["clips/OAF_thought_happy.wav"])
    for i in happy_list:
        create(i,sizedlist[i],'happy',gpt_cond_latent,speaker_embedding)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["clips/OAF_thought_angry.wav"])
    for i in anger_list:
        create(i,sizedlist[i],'angry',gpt_cond_latent,speaker_embedding)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["clips/OAF_thought_sad.wav"])
    for i in sad_list:
        create(i,sizedlist[i],'sad',gpt_cond_latent,speaker_embedding)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["clips/OAF_thought_neutral.wav"])
    for i in neutral_list:
        create(i,sizedlist[i],'uio',gpt_cond_latent,speaker_embedding)
    
    
    # for i,j in enumerate(final_list):
    #      if(j[0]!=''):
    #         create(i,j[0],j[1])
    combined_audio = AudioSegment.silent()
    for i in range(l):
        segment =AudioSegment.from_file(f'xtts_streaming{i}.wav',format='wav')
        combined_audio += segment 
    combined_audio.export('true.wav',format='wav')    
    return StreamingResponse(open("/Users/manasagarwal/Documents/working_model/true.wav", "rb"), media_type="audio/mpeg", headers={"Content-Disposition": "attachment;filename=result.mp3"})
@app.get("/")
def read_hello():
    return {"message": "Hello, World!"}
@app.post("/convert_pdf_to_mp3/")
async def convert_pdf_to_mp3(file: UploadFile = File(...)):
    model.load_checkpoint(config, checkpoint_dir="", use_deepspeed=False)
    temp_pdf_path = "temp_pdf.pdf"
    file=await file.read()
    with open(temp_pdf_path, "wb") as temp_pdf_file:
        temp_pdf_file.write(file)
    print("got it")
    pdf_reader = PyPDF2.PdfReader(temp_pdf_path)
    text = ''
    if pdf_reader.pages:
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    await tex_to_wav(text)
    wav_content = read_wav_file()
    return StreamingResponse(io.BytesIO(wav_content), media_type="audio/wav", headers={"Content-Disposition": "attachment;filename=result.wav"})
