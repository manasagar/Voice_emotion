import requests
api_url = "http://127.0.0.1:8000/convert_pdf_to_mp3/"
pdf_file_path = r"/Users/manasagarwal/Documents/working_model/manas.pdf"
files = {'file': open(pdf_file_path, 'rb')}
print(files)
def string_to_bits(input_string):
    return bytes(input_string, 'utf-8')

response = requests.post(api_url, files=files)
with open(f"audio.wav","wb") as file:
        file.write(response.content)

