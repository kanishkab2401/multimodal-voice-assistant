import asyncio
import json
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from services.vad import SimpleVAD
from services.asr import transcribe
from services.llm import stream_response
from services.tts import stream_speech
from services.aggregator import SentenceAggregator


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def send_event(ws: WebSocket, event_type: str, **data):
    """
    Every event the backend emits goes through this one function, in one
    schema: {type, ts, ...payload}. This is the hook Phase 2's latency
    dashboard will read from — keep every timing signal flowing through
    here rather than ad-hoc prints.
    """
    await ws.send_text(json.dumps({"type": event_type, "ts": time.time(), **data}))


@app.websocket("/ws")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    vad = SimpleVAD()
    pcm_buffer = bytearray()
    history: list = []

    await send_event(ws, "session.ready")

    try:
        while True:
            msg = await ws.receive()

            if msg.get("bytes") is not None:
                chunk = msg["bytes"]
                pcm_buffer.extend(chunk)
                utterance_done = vad.process_chunk(chunk)

                if utterance_done and len(pcm_buffer) > 0:
                    audio_data = bytes(pcm_buffer)
                    pcm_buffer.clear()
                    # Fire-and-forget so the receive loop keeps accepting
                    # audio for the *next* utterance while this one processes.
                    asyncio.create_task(handle_utterance(ws, audio_data, history))

            elif msg.get("text") is not None:
                control = json.loads(msg["text"])
                if control.get("type") == "reset":
                    pcm_buffer.clear()
                    history.clear()
                    vad.reset()
                    await send_event(ws, "session.reset")

    except WebSocketDisconnect:
        pass


async def handle_utterance(ws: WebSocket, audio_data: bytes, history: list):
    t_start = time.time()
    loop = asyncio.get_event_loop()

    # --- ASR ---
    await send_event(ws, "asr.started")
    try:
        transcript = await loop.run_in_executor(None, transcribe, audio_data)
    except Exception as e:
        await send_event(ws, "error", stage="asr", message=str(e))
        return

    t_asr_done = time.time()
    if not transcript:
        await send_event(ws, "asr.empty")
        return

    await send_event(
        ws, "transcript.final",
        text=transcript,
        latency_ms=int((t_asr_done - t_start) * 1000),
    )

    # --- LLM (streaming, with the model deciding for itself whether it
    # needs to search) + TTS (streamed per-sentence as tokens arrive) ---
    aggregator = SentenceAggregator()
    full_response = ""
    first_token_seen = False

    try:
        async for event in stream_response(transcript, history):
            if event["type"] == "search_started":
                await send_event(ws, "search.started", query=event["query"])

            elif event["type"] == "search_done":
                await send_event(ws, "search.done", found=event["found"])

            elif event["type"] == "token":
                if not first_token_seen:
                    first_token_seen = True
                    await send_event(
                        ws, "llm.first_token",
                        latency_ms=int((time.time() - t_asr_done) * 1000),
                    )
                full_response += event["text"]
                await send_event(ws, "llm.token", text=event["text"])

                for sentence in aggregator.feed(event["text"]):
                    await speak_sentence(ws, sentence, t_asr_done)

        for sentence in aggregator.flush():
            await speak_sentence(ws, sentence, t_asr_done)

    except Exception as e:
        await send_event(ws, "error", stage="llm", message=str(e))
        return

    history.append({"role": "user", "content": transcript})
    history.append({"role": "assistant", "content": full_response})

    await send_event(
        ws, "turn.complete",
        response_text=full_response,
        total_latency_ms=int((time.time() - t_start) * 1000),
    )


async def speak_sentence(ws: WebSocket, sentence: str, t_ref: float):
    if not sentence.strip():
        return

    await send_event(ws, "tts.sentence_start", text=sentence)
    first_chunk = True
    try:
        async for audio_chunk in stream_speech(sentence):
            if first_chunk:
                first_chunk = False
                await send_event(
                    ws, "tts.first_byte",
                    latency_ms=int((time.time() - t_ref) * 1000),
                )
            await ws.send_bytes(audio_chunk)
    except Exception as e:
        await send_event(ws, "error", stage="tts", message=str(e))
        return

    await send_event(ws, "tts.sentence_end")


@app.get("/")
def health():
    return {"status": "ok"}
