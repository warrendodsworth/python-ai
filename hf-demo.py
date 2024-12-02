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
import streamlit as st


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

    # Load and play the audio file
    # Audio("speech.wav", autoplay=True) # doesn't autoplay
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

# TEST
# scenario = img2text("photo.jpg")
# story = generate_story(scenario)
# story = "The church clock chimed midnight, its echo swallowed by the silent town."
# text2speech(story)


def main():
    st.set_page_config(page_title="Image to story")
    st.header("Turn photos into an audio stories")
    uploaded_file = st.file_uploader("Choose a photo..", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded photo", use_container_width=True)
        scenario = img2text(uploaded_file.name)
        st.header("Scenario")
        st.write(scenario)

        story = generate_story(scenario)
        st.header("Story")
        st.write(story)

        text2speech(story)
        st.header("Audio")
        st.audio("speech.wav")


if __name__ == "__main__":
    main()
