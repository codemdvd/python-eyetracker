# src/eyetrk/web_bridge/server.py

from __future__ import annotations

import sys
import asyncio
import queue
from typing import Dict, Any

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel

from eyetrk.core.types import Sample

# Регистрируем адаптеры (WebGazerAdapter и т.п.) из CLI
adapters_registry: Dict[str, Any] = {}

# Очереди исходящих событий (calib_point_start, task_start и т.п.) по трекеру
events_queues: Dict[str, "queue.Queue[dict]"] = {}

# Просто счётчик принятых сэмплов для отладки
recv_counts: Dict[str, int] = {}

app = FastAPI()


class WGSample(BaseModel):
    tracker_id: str
    session_id: str
    timestamp_ms: int
    x_norm: float | None = None
    y_norm: float | None = None
    confidence: float | None = None
    validity: int = 0
    stim_id: str | None = None
    event: str | None = None


@app.get("/health")
def health():
    return {"ok": True}


def queue_event(tracker_id: str, event: dict) -> None:
    """
    Кладём событие в очередь для браузера.
    Можно вызывать из Orchestrator._broadcast(...) для синхронной калибровки.
    """
    q = events_queues.setdefault(tracker_id, queue.Queue())
    q.put(event)


@app.websocket("/ws/{tracker_id}")
async def ws_endpoint(ws: WebSocket, tracker_id: str):
    """
    WebSocket-бридж:
    - принимает JSON-сэмплы из браузера и прокидывает их в адаптер (WebGazerAdapter.emit)
    - отправляет в браузер события из queue_event(...)
    """
    print(f"[web-bridge] WS connect: {tracker_id}", file=sys.stderr, flush=True)
    recv_counts[tracker_id] = 0
    await ws.accept()

    # Очередь для исходящих событий в браузер
    q: "queue.Queue[dict]" = events_queues.setdefault(tracker_id, queue.Queue())

    async def sender():
        """
        Отправляет события из q в браузер.
        Бежит в отдельной задаче, пока соединение живо.
        """
        try:
            while True:
                try:
                    msg = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    await ws.send_json(msg)
                except Exception as e:
                    print(f"[web-bridge] WS send error for {tracker_id}: {e}", file=sys.stderr, flush=True)
                    break
        except asyncio.CancelledError:
            # Нормальное завершение, когда нас отменили из finally
            pass
        finally:
            # По завершении чистим очередь для этого трекера
            events_queues.pop(tracker_id, None)

    sender_task = asyncio.create_task(sender())

    try:
        while True:
            data = await ws.receive_json()
            s = WGSample(**data)

            ad = adapters_registry.get(tracker_id)
            if ad is not None:
                # Оборачиваем WGSample в общий Sample и отправляем в адаптер
                sample = Sample(
                    session_id=s.session_id,
                    tracker_id=s.tracker_id,
                    timestamp_ms=s.timestamp_ms,
                    x_norm=s.x_norm,
                    y_norm=s.y_norm,
                    confidence=s.confidence,
                    validity=s.validity,
                    stim_id=s.stim_id,
                    event=s.event,
                )
                try:
                    ad.emit(sample)  # WebGazerAdapter.emit
                except Exception as e:
                    print(f"[web-bridge] adapter error for {tracker_id}: {e}", file=sys.stderr, flush=True)

            recv_counts[tracker_id] += 1
            cnt = recv_counts[tracker_id]
            if cnt <= 5 or cnt % 100 == 0:
                print(f"[web-bridge] recv {tracker_id}: {cnt}", file=sys.stderr, flush=True)

    except WebSocketDisconnect:
        print(f"[web-bridge] WS disconnect: {tracker_id}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[web-bridge] WS error for {tracker_id}: {e}", file=sys.stderr, flush=True)
    finally:
        # Останавливаем отправитель и аккуратно ждём его завершения
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            # Ожидаемо, мы сами его отменили
            pass
        except Exception:
            # Любые другие проблемы нам не критичны
            pass
