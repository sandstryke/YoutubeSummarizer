from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import YoutubeLoader
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time
import logging
import os
from starlette.requests import Request

# Initialize FastAPI app
app = FastAPI()

# Initialize Jinja2 templates for FastAPI (equivalent to Flask's render_template)
templates = Jinja2Templates(directory="chatbot/templates")

# Initialize the OllamaLLM model (we'll use "llama3" as the model here)
llm = OllamaLLM(model="llama3")

# Define a PromptTemplate to generate dynamic prompts for summarization
product_description_template = PromptTemplate(
    input_variables=["video_transcript"],
    template="""Read through the entire transcript carefully.
    Provide a concise summary of the video's main topic and purpose.
    Extract and list the five most interesting or important points from the transcript. 
    For each point: State the key idea in a clear and concise manner.

    - Ensure your summary and key points capture the essence of the video without including unnecessary details.
    - Use clear, engaging language that is accessible to a general audience.
    - If the transcript includes any statistical data, expert opinions, or unique insights, 
      prioritize including these in your summary or key points.

    Video transcript: {video_transcript}
    """
)

# Create an LLMChain with the prompt template and the LLM
chain = LLMChain(llm=llm, prompt=product_description_template)

# Function to load video and summarize it
def load_video_with_timeout(video_url, timeout=30):
    start_time = time.time()
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    try:
        data = loader.load()
        if time.time() - start_time > timeout:
            return None, "Loading took too long, aborting."
        logging.info(f"Successfully loaded video: {video_url}")
        return data, None
    except Exception as e:
        logging.error(f"Error loading video: {str(e)}")
        return None, f"Error loading video: {str(e)}"

# Function to summarize the video
def summarize_youtube_video(video_url):
    logging.info(f"Starting to summarize video: {video_url}")
    data, error = load_video_with_timeout(video_url)
    
    if error:
        logging.error(f"Error in loading or summarizing: {error}")
        return error

    video_text = "\n".join([doc.page_content for doc in data])  # Assuming data is a list of documents
    logging.info(f"Video transcript loaded. Starting summary generation...")
    
    summary = chain.invoke({"video_transcript": video_text})
    
    logging.info("Summary generated successfully.")
    return summary['text']

# Route to the home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "summary": "", "error": ""})

@app.post("/", response_class=HTMLResponse)
async def get_summary(request: Request, video_url: str = Form(...)):
    # Validate the YouTube URL
    if "youtube.com" not in video_url:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid YouTube URL. Please try again.", "summary": ""})
    
    # Summarize the video
    summary = summarize_youtube_video(video_url)

    return templates.TemplateResponse("index.html", {"request": request, "summary": summary, "error": ""})

# Run the app with Uvicorn (run this in the terminal)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5003)








