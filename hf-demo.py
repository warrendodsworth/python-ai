from transformers import pipeline
from dotenv import find_dotenv, load_dotenv
import torch

load_dotenv(find_dotenv())

# img2text
def img2text(url):
  image_2_text = pipeline('image-to-text', model="Salesforce/blip-image-captioning-base")
  text = image_2_text(url)[0]['generated_text']
  print(text)
  return text

scenario = img2text('photo.jpg')

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

story = generate_story("a church in the shadows")