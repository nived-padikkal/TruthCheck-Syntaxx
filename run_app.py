import sys, traceback
try:
    import inference_2
    print("inference_2 import ok", flush=True)
    import gradio as gr
    print("gradio import ok", flush=True)
    video_interface = gr.Interface(inference_2.deepfakes_video_predict, gr.Video(), "text",
                                   examples=["videos/aaa.mp4"], cache_examples=False)
    image_interface = gr.Interface(inference_2.deepfakes_image_predict, gr.Image(), "text",
                                   examples=["images/lady.jpg"], cache_examples=False)
    audio_interface = gr.Interface(inference_2.deepfakes_spec_predict, gr.Audio(), "text",
                                   examples=["audios/DF_E_2000027.flac"], cache_examples=False)
    app = gr.TabbedInterface(interface_list=[image_interface, video_interface, audio_interface],
                             tab_names=['Image inference', 'Video inference', 'Audio inference'])
    print("launching", flush=True)
    app.launch(share=False)
except Exception:
    traceback.print_exc()
    sys.exit(1)
