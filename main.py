import sys
import os
from langchain.chat_models import ChatOpenAI
from pydub import AudioSegment
import openai
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

load_dotenv()  # take environment variables from .env.

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key


audio_path = './assets/how-to-get-rich.mp3'


def check_file_size(audio_path):
  print('Chunking audio file...')
  file_name = os.path.basename(audio_path)
  file_size = os.path.getsize(audio_path)
  print(f'↪ File name: {file_name}')
  print(f'↪ File size: {file_size} bytes')

  # Get length of audio file
  audio = AudioSegment.from_mp3(audio_path)
  duration = audio.duration_seconds
  print(f'↪ File duration: {duration} seconds')

  if file_size > 25 * 1024 * 1024:
    sys.exit('↪ File size is too large!')
  else:
    print('↪ File size is ok!')


def transcribe(audio_path):
  print('Transcribing audio file...')

  audio_name = os.path.splitext(os.path.basename(audio_path))[0]
  print(f'↪ Audio name: {audio_name}')

  transcript_path = f"./transcripts/{audio_name}.txt"
  print(f'↪ Transcript path: {transcript_path}')

  full_transcript = ''

  if not os.path.exists(transcript_path):
    print('↪ Need to transcribe file')

    # Convert the MP3 file to text using Whisper API
    file = open(audio_path, "rb")

    print('↪ Transcribing audio file...')
    response = openai.Audio.transcribe("whisper-1", file)

    # Check for errors in the API response
    if "error" in response:
      error_msg = response["error"]["message"]  # type: ignore
      raise Exception(f"⚠️ Transcription error: {error_msg}")

    # # Extract the transcript from the API response
    full_transcript = response["text"].strip()  # type: ignore

    # # Save the transcript to a text file
    with open(transcript_path, "w") as f:
      f.write(full_transcript)
      print(
          f"\t\t↪ saved transcript to {transcript_path} (words: {len(full_transcript.split())}")
  else:
    print('↪ Transcription already exists!')
    # Load the transcript from the text file
    with open(transcript_path, "r") as f:
      full_transcript += f.read()

  print(
      f'↪ Total words: {len(full_transcript.split())} -- characters: {len(full_transcript)}')

  return full_transcript


def summarize(transcript):
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                   openai_api_key=openai_api_key)  # type: ignore

  texts = CharacterTextSplitter().split_text(transcript)
  docs = [Document(page_content=t) for t in texts[:3]]

  chain = load_summarize_chain(llm, chain_type="map_reduce")
  return chain.run(docs)

# Main application


def main():
  print('Initializing summarizer...')
  print(f'Audio path: {audio_path} ')

  print(f'🗣️  Initializing Whisper transcriber...')
  check_file_size(audio_path)
  transcript = transcribe(audio_path)
  summary = summarize(transcript)
  print(f"📝 Summary: {summary}")


# ## execute main
if __name__ == "__main__":
  main()
