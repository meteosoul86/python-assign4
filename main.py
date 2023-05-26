from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import gradio as gr


tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")


def predict(input, history=[]):
    instruction = 'Instruction: given a dialog context, you need to response empathically'
    knowledge = '  '
    s = list(sum(history, ()))
    s.append(input)
    dialog = ' EOS '.join(s)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    top_p = 0.9
    min_length = 8
    max_length = 64
    new_user_input_ids = tokenizer.encode(f"{query}", return_tensors='pt')

    output = model.generate(new_user_input_ids, min_length=int(
        min_length), max_length=int(max_length), top_p=top_p, do_sample=True).tolist()

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    history.append((input, response))

    return history, history


gr.Interface(fn=predict,
             inputs=["text",'state'],
             outputs=["chatbot",'state']).launch(debug = True, share = True)