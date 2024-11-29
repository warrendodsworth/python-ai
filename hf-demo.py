from transformers import pipeline
from dotenv import find_dotenv, load_dotenv
import torch
import requests
import os
from IPython.display import Audio


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# img2text
def img2text(url):
	image_2_text = pipeline('image-to-text', model="Salesforce/blip-image-captioning-base")
	text = image_2_text(url)[0]['generated_text']
	print(text)
	return text


# text to speech - to generate the story
def generate_story(scenario):
	pipe = pipeline(
			"text-generation",
			model="google/gemma-2-2b-it",
			model_kwargs={"torch_dtype": torch.bfloat16},
			device_map="auto",  # cuda or replace with "mps" to run on a Mac device
	)
	content = f"""
		You are a storyteller. You can generate a short story based on a simple narrative. Your story should be no more than 20 words.
		CONTEXT: {scenario}
	"""
	messages = [
			{"role": "user", "content": content},
	]

	outputs = pipe(messages, max_new_tokens=256)
	assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
	print(assistant_response)
	return assistant_response



# text to speech - to read the story
def text2speech(text: str):
	API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
	headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
	def query(payload):
		response = requests.post(API_URL, headers=headers, json=payload)
		return response.json()
		
	audio, sampling_rate = query({
		"inputs": "The answer to the universe is 42",
	})
	print('SAMPLING: ',sampling_rate)
	# You can access the audio with IPython.display for example
	Audio(audio, rate=sampling_rate)


# scenario = img2text('photo.jpg')
# story = generate_story(scenario)
demo_story = "The church clock chimed midnight, its echo swallowed by the silent town."
text2speech(demo_story)
