import io
from transformers import pipeline
from dotenv import find_dotenv, load_dotenv
import torch
import requests
import os
from IPython.display import Audio
from datasets import load_dataset
import soundfile as sf
import simpleaudio as sa


# import warnings
# from urllib3.exceptions import NotOpenSSLWarning
# warnings.simplefilter("ignore", NotOpenSSLWarning)

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# img2text
def img2text(url):
    image_2_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base"
    )
    text = image_2_text(url, max_new_tokens=256)[0]["generated_text"]
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
    synthesiser = pipeline("text-to-speech", model="microsoft/speecht5_tts")
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # You can replace this embedding with your own as well.
    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    # Convert audio to file
    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
    Audio("speech.wav", autoplay=True)

    # Load and play the audio file
    wave_obj = sa.WaveObject.from_wave_file("speech.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until playback is finished


# # Convert audio data to a byte stream
# audio_buffer = io.BytesIO()
# sf.write(
#     audio_buffer, speech["audio"], samplerate=speech["sampling_rate"], format="WAV"
# )
# audio_buffer.seek(0)  # Rewind the buffer to the beginning

# # Play audio from the byte stream
# Audio(audio_buffer.read(), rate=speech["sampling_rate"], autoplay=True)


# scenario = img2text('photo.jpg')
# story = generate_story(scenario)
story = "The church clock chimed midnight, its echo swallowed by the silent town."
text2speech(story)
