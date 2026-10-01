# Multilingual YouTube Video Summarizer with Llama 3

A Python application for extracting YouTube transcripts and generating concise summaries with **Llama 3**. The project combines transcript processing, chunking, local LLM inference, and translation to make long-form video content easier to review across multiple languages.

## What It Does

- Accepts a YouTube video URL and retrieves video information.
- Extracts the available transcript from the video.
- Splits long transcripts into manageable chunks for LLM processing.
- Uses **Llama 3 through Ollama** to summarize transcript content.
- Uses a map-reduce summarization workflow for longer videos.
- Supports multilingual output through translation.
- Provides configurable temperature, chunk-size, and overlap settings.
- Includes a browser-based interface for interacting with the workflow.

## Tech Stack

- **Python**
- **Llama 3**
- **Ollama**
- **LangChain**
- **Gradio**
- **YouTube transcript/document loaders**
- **deep-translator**

## Workflow

```text
YouTube URL
    ↓
Video metadata + transcript
    ↓
Transcript cleaning / chunking
    ↓
Llama 3 summarization
    ↓
Map-reduce combination
    ↓
Optional translation
    ↓
Multilingual summary
```

## Why This Project

Long videos can contain useful information but are time-consuming to review. This project explores how a locally served large language model can turn video transcripts into shorter, readable summaries while allowing the user to request output in different languages.

It also provided hands-on experience with LLM application workflows, prompt-based summarization, text chunking, local model serving, and integrating multiple Python libraries into one application.

## Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/sisbeyene/MULTI-LINGUAL-YOUTUBEVIDEO-SUMMARIZER-USING-LLAMA3.git
cd MULTI-LINGUAL-YOUTUBEVIDEO-SUMMARIZER-USING-LLAMA3
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and run Ollama

Install Ollama and make sure Llama 3 is available locally.

```bash
ollama pull llama3
```

Start Ollama if it is not already running, then verify the local service is available before launching the application.

### 4. Run the application

```bash
python main.py
```

Open the local URL shown by the application in your browser.

## Using the Application

1. Paste a YouTube URL.
2. Retrieve the video information and transcript.
3. Select the desired output language.
4. Adjust summarization settings if needed.
5. Generate the summary.

## Multilingual Support

The workflow supports output across a broad set of languages, including English, Spanish, French, German, Arabic, Amharic, Oromo, Somali, Swahili, Hindi, Bengali, Chinese, Japanese, Korean, and others supported by the translation component.

## Repository Structure

```text
.
├── main.py              # Main application and summarization workflow
├── requirements.txt     # Python dependencies
├── Untitled.ipynb       # Development notebook
├── Untitled1.ipynb      # Development/experimentation notebook
└── README.md
```

## Areas for Improvement

- Improve transcript error handling when captions are unavailable.
- Add automated tests for the transcript and summarization pipeline.
- Evaluate summary quality across languages and video lengths.
- Refactor experimental notebook work into clearer modules.
- Add a hosted demo or application screenshots.

## Project Note

This repository represents an LLM application project built around existing open-source tools and models including Llama 3, Ollama, LangChain, and Gradio. The focus is on integrating these components into an end-to-end multilingual video summarization workflow.
