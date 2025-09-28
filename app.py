import gradio as gr
from duckduckgo_search import DDGS
from transformers import pipeline

# Usamos un modelo de texto pequeño para que funcione en CPU
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

def buscar_responder(pregunta):
    # Buscar info en la web
    with DDGS() as ddgs:
        results = ddgs.text(pregunta, max_results=3)
        context = "\n".join([r["body"] for r in results])

    # Construir prompt
    prompt = f"Eres un técnico experto en reparación de electrodomésticos.\nPregunta: {pregunta}\nInformación encontrada:\n{context}\nRespuesta clara y técnica:"
    respuesta = qa_model(prompt, max_new_tokens=200)[0]["generated_text"]

    return respuesta

# Interfaz con Gradio
iface = gr.Interface(
    fn=buscar_responder,
    inputs="text",
    outputs="text",
    title="Chatbot Técnico en Electrónica y Electrodomésticos",
    description="Haz tu consulta y obtén una respuesta técnica paso a paso."
)

iface.launch()
