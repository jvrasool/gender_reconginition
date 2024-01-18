import gradio as gr
import pandas as pd
import numpy as np
import test

# Create a Gradio app
def gender_recoginition(test_example):
    result = getattr(test, "test_sample")(test_example)
    print(result)
    return result

demo = gr.Interface(
    fn = gender_recoginition,
    inputs = gr.Textbox(lines=2, placeholder="Enter path for test example..."),
    outputs = "text"
)
demo.launch()