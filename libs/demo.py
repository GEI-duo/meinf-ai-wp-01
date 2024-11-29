import gradio as gr


def run_frontend() -> None:

    with gr.Blocks() as demo:
        text_box = gr.Textbox(
            label="Enter your name",
            placeholder="Type your name here",
            lines=1,
        )

    demo.launch()


if __name__ == "__main__":
    run_frontend()
