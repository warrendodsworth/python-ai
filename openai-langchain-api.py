# llm

from langchain import PromptTemplate, LLMChain, OpenAI
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

def generate_story(scenario):
  template: f"""
    You are a storyteller. You can generate a short story based on a simple narrative. Your story should be no more than 20 words.

    CONTEXT: {scenario}
    STORY:
  """
  prompt = PromptTemplate(template=template, input_variables="[scenario]")
  story_llm = LLMChain(llm=OpenAI(
    model_name='gpt-3.5-turbo', temperature=1), prompt=prompt, verbose=True)

  story = story_llm.predict(scenario=scenario)
  print(story)
  return story

story = generate_story("a church in the shadows")
