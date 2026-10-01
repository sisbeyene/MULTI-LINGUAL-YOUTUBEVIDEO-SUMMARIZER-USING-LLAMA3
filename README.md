# Multilingual YouTube Video Summarizer with Llama 3

A Python application that extracts YouTube transcripts and turns long-form video content into readable summaries using a **locally served Llama 3 model**. The application also supports translated output for multilingual use.

## Features

- Retrieves YouTube video information and available transcripts
- Estimates transcript token count
- Splits long transcripts into manageable chunks
- Summarizes content with **Llama 3 via Ollama**
- Uses a map-reduce workflow for long videos
- Supports multilingual summary output
- Provides configurable temperature, chunk size, and overlap
- Includes an interactive **Gradio** interface
- Handles invalid URLs, unavailable captions, translation failures, and local-model errors more clearly

## Tech Stack

**Python • Llama 3 • Ollama • LangChain • Gradio • tiktoken • deep-translator**

## How It Works

```text
YouTube URL
    ↓
Video metadata + transcript
    ↓
Validation and token estimation
    ↓
Transcript chunking
    ↓
Llama 3 map summaries
    ↓
Map-reduce combination
    ↓
Optional translation
    ↓
Final summary
```

## Why I Built It

Long videos often contain useful information that takes significant time to review. This project explores how a locally hosted large language model can make that content easier to consume while keeping the core LLM inference workflow on the user's machine.

The project provided hands-on experience with LLM application development, prompt design, text chunking, map-reduce summarization, local model serving, translation, and interactive application development.

## Getting Started

### Prerequisites

- Python 3
- Ollama installed locally
- A YouTube video with accessible captions/transcript

### 1. Clone the repository

```bash
git clone https://github.com/sisbeyene/MULTI-LINGUAL-YOUTUBEVIDEO-SUMMARIZER-USING-LLAMA3.git
cd MULTI-LINGUAL-YOUTUBEVIDEO-SUMMARIZER-USING-LLAMA3
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Llama 3 with Ollama

```bash
ollama pull llama3
```

Make sure Ollama is running locally before starting the application.

### 4. Launch the app

```bash
python main.py
```

Then open the local Gradio address displayed in the terminal.

## Using the Application

1. Paste a valid YouTube URL.
2. Click **Get Info** to retrieve the title and description.
3. Click **Get Transcript** to inspect the transcript and estimated token count.
4. Choose the summarization settings and output language.
5. Click **Generate Summary**.

## Current Repository Structure

```text
.
├── main.py              # Application, transcript processing, and LLM workflow
├── requirements.txt     # Python dependencies
├── .gitignore           # Local/environment files excluded from version control
├── Untitled.ipynb       # Early setup notebook (legacy)
├── Untitled1.ipynb      # Development/experimentation notebook (legacy)
└── README.md
```

The two notebooks are retained as development history; `main.py` is the maintained application entry point.

## Current Limitations

- Requires Ollama and Llama 3 to be running locally
- Depends on an accessible YouTube transcript/captions
- Translation uses an external translation library/service
- Summary quality varies with transcript quality, video length, and model behavior
- The project does not yet include a formal summary-quality benchmark

## Next Improvements

- Add automated tests for URL validation, chunking, and transcript handling
- Evaluate summary quality across different video categories and lengths
- Add application screenshots or a short demo
- Separate the application into smaller modules as the project grows
- Add optional export of summaries to Markdown or text

## Project Note

This project integrates open-source models and libraries—including Llama 3, Ollama, LangChain, and Gradio—into an end-to-end video summarization application. The project focus is the design and integration of the workflow rather than training a language model from scratch.
