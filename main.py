"""Multilingual YouTube summarizer powered by a locally served Llama 3 model."""

import re

import gradio as gr
import pytube
import requests
import tiktoken
from deep_translator import GoogleTranslator
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import Ollama

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3"
REQUEST_TIMEOUT = 15

LANGUAGES = [
    "Amharic", "Arabic", "Akan", "Bengali", "Bhojpuri", "Bulgarian", "Catalan",
    "Chinese (Simplified)", "Croatian", "Czech", "Danish", "Dutch", "English",
    "Estonian", "Filipino", "Finnish", "French", "German", "Greek", "Gujarati",
    "Hausa", "Hebrew", "Hindi", "Hungarian", "Icelandic", "Igbo", "Indonesian",
    "Italian", "Japanese", "Kannada", "Kinyarwanda", "Korean", "Latvian",
    "Lithuanian", "Luo", "Malay", "Malayalam", "Marathi", "Nepali", "Norwegian",
    "Odia", "Oromo", "Persian", "Polish", "Portuguese", "Punjabi", "Romanian",
    "Russian", "Slovak", "Slovenian", "Somali", "Spanish", "Swahili", "Swedish",
    "Tamil", "Telugu", "Thai", "Tigrinya", "Turkish", "Twi", "Ukrainian",
    "Urdu", "Vietnamese", "Welsh", "Wolof", "Xhosa", "Yoruba", "Zulu",
]

MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following transcript chunk accurately and clearly. Preserve the "
        "main ideas, important supporting details, and conclusions. Avoid adding facts "
        "that are not present in the source.\n\nTranscript:\n{text}"
    ),
)

COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "Combine the following partial summaries into one coherent final summary. "
        "Remove repetition while preserving the important ideas, details, and conclusions.\n\n"
        "Partial summaries:\n{text}"
    ),
)


def validate_youtube_url(url: str) -> str:
    """Perform a basic validation before sending a URL to YouTube-related libraries."""
    url = (url or "").strip()
    if not url:
        raise gr.Error("Please enter a YouTube URL.")
    if not re.match(r"^https?://(www\.)?(youtube\.com|youtu\.be)/", url):
        raise gr.Error("Please enter a valid YouTube URL.")
    return url


def get_youtube_description(url: str) -> str:
    """Extract the short video description from the public YouTube page."""
    url = validate_youtube_url(url)
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        match = re.search(r'"shortDescription":"((?:\\.|[^"\\])*)"', response.text)
        if not match:
            return "Description unavailable."
        return bytes(match.group(1), "utf-8").decode("unicode_escape")
    except requests.RequestException:
        return "Description unavailable."


def get_youtube_info(url: str):
    """Return a video's title and description."""
    url = validate_youtube_url(url)
    try:
        video = pytube.YouTube(url)
        return video.title or "Title unavailable.", get_youtube_description(url)
    except Exception as exc:
        raise gr.Error(f"Unable to retrieve video information: {exc}") from exc


def load_transcript(url: str):
    """Load the transcript as LangChain documents."""
    url = validate_youtube_url(url)
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        docs = loader.load()
        if not docs:
            raise ValueError("No transcript was returned.")
        return docs
    except Exception as exc:
        raise gr.Error(
            "Unable to retrieve a transcript. The video may not have accessible captions."
        ) from exc


def transcript_to_text(docs) -> str:
    return " ".join(doc.page_content for doc in docs).strip()


def get_youtube_transcription(url: str):
    """Return transcript text and an estimated token count."""
    text = transcript_to_text(load_transcript(url))
    encoding = tiktoken.get_encoding("cl100k_base")
    return text, len(encoding.encode(text))


def get_text_splitter(chunk_size: int, overlap_size: int):
    chunk_size = int(chunk_size)
    overlap_size = int(overlap_size)
    if overlap_size >= chunk_size:
        raise gr.Error("Overlap size must be smaller than chunk size.")
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    )


def translate_summary(summary: str, language: str) -> str:
    """Translate a generated English summary when another language is selected."""
    if language == "English":
        return summary
    try:
        return GoogleTranslator(source="auto", target=language.lower()).translate(summary)
    except Exception as exc:
        raise gr.Error(f"Summary generated, but translation to {language} failed.") from exc


def get_transcription_summary(
    url: str,
    temperature: float,
    chunk_size: int,
    overlap_size: int,
    language: str,
):
    """Summarize a YouTube transcript with a map-reduce Llama 3 workflow."""
    docs = load_transcript(url)
    splitter = get_text_splitter(chunk_size, overlap_size)
    split_docs = splitter.split_documents(docs)

    llm = Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=float(temperature),
    )
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
    )

    try:
        output = chain.invoke(split_docs)
        summary = output["output_text"].strip()
    except Exception as exc:
        raise gr.Error(
            "Summarization failed. Make sure Ollama is running and the llama3 model is installed."
        ) from exc

    return translate_summary(summary, language)


CUSTOM_CSS = """
.gr-panel {
    border-radius: 8px;
    padding: 20px;
}
"""


def build_interface():
    """Build and return the Gradio application."""
    with gr.Blocks(css=CUSTOM_CSS, title="Multilingual YouTube Summarizer") as demo:
        gr.Markdown(
            "# Multilingual YouTube Summarizer with Llama 3\n"
            "Extract a transcript, estimate its size, and create a multilingual summary "
            "using a locally served Llama 3 model."
        )

        with gr.Row():
            url = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                scale=4,
            )
            get_info_button = gr.Button("Get Info", variant="primary", scale=1)
            clear_button = gr.ClearButton()

        with gr.Row():
            title = gr.Textbox(label="Title", interactive=False)
            description = gr.Textbox(label="Description", lines=3, interactive=False)

        with gr.Row():
            transcript_button = gr.Button("Get Transcript", variant="primary")
            summarize_button = gr.Button("Generate Summary", variant="primary")

        with gr.Row():
            temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
            chunk_size = gr.Number(value=5000, minimum=200, step=100, label="Chunk Size")
            overlap_size = gr.Number(value=100, minimum=0, step=10, label="Overlap Size")
            language = gr.Dropdown(LANGUAGES, value="English", label="Output Language")

        token_count = gr.Number(label="Estimated Token Count", interactive=False)

        with gr.Row():
            transcript = gr.Textbox(label="Transcript", lines=15, show_copy_button=True)
            summary = gr.Textbox(label="Summary", lines=15, show_copy_button=True)

        get_info_button.click(get_youtube_info, inputs=url, outputs=[title, description])
        transcript_button.click(
            get_youtube_transcription,
            inputs=url,
            outputs=[transcript, token_count],
        )
        summarize_button.click(
            get_transcription_summary,
            inputs=[url, temperature, chunk_size, overlap_size, language],
            outputs=summary,
        )
        clear_button.add([url, title, description, transcript, summary, token_count])

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="localhost", server_port=7860)
