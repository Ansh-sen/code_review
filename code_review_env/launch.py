#!/usr/bin/env python3
"""Launch script — serves Gradio UI + FastAPI REST API on port 7860."""

import gradio as gr
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from server.app import app as fastapi_app
from gradio_demo import demo

# Mount Gradio on the root path of the FastAPI app
gr.mount_gradio_app(fastapi_app, demo, path="/")

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)
