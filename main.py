from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ollama import chat

app = FastAPI(title="Company AI Assistant")

templates = Jinja2Templates(directory="templates")

KNOWLEDGE_FILE = "company.txt"
knowledge = Path(KNOWLEDGE_FILE).read_text(encoding="utf-8")

MODEL = "huihui_ai/lfm2.5-abliterated"

history = []

def now():
    return datetime.now().strftime("%I:%M %p")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"history": history}
    )


@app.post("/", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):

    history.append({"role": "user", "content": question, "time": now()})

    try:
        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful and intelligent company assistant. "
                        "You have access to the company's information and should prioritize it when answering company-related questions, however you are allowed to give your own opinions. "
                        "However, you are also free to think, reason, and answer general questions using your own knowledge. "
                        "If a question is about the company but the answer is not in the provided information, "
                        "be honest and say you are not sure about that specific detail, but still try to help as best you can."
                    )
                },
                {
                    "role": "user",
                    "content": f"COMPANY INFORMATION:\n{knowledge}\n\nQUESTION:\n{question}"
                }
            ]
        )
        answer = response["message"]["content"]
        history.append({"role": "bot", "content": answer, "time": now()})

    except Exception as e:
        history.append({"role": "error", "content": str(e), "time": now()})

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"history": history}
    )


@app.get("/clear", response_class=HTMLResponse)
async def clear_chat(request: Request):
    history.clear()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"history": history}
    )