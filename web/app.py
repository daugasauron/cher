import os
import json
import time
import secrets
import worker as w
import struct
import plotly
import asyncio
import uvicorn
import subprocess
import mojo.importer
from itertools import count
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from concurrent.futures import ProcessPoolExecutor
from cher_mojo import Network

STATIC_DIR_NAME = 'static'

counter = count(0)

async def worker_sender(
        message_queue: asyncio.Queue[tuple[str, str]],
        writer: asyncio.StreamWriter,
):
    while True:
        msg_type, msg_value = await message_queue.get()
        print(f'API -> worker: sending {msg_type}: {msg_value}')

        if msg_type == 'event' and msg_value == 'start':
            writer.write(struct.pack('!h', w.MessageReceive.START))
        elif msg_type == 'event' and msg_value == 'stop':
            writer.write(struct.pack('!h', w.MessageReceive.STOP))
        elif msg_type == 'event' and msg_value == 'generate_path':
            writer.write(struct.pack('!h', w.MessageReceive.GENERATE_TEST_PATH))
        elif msg_type == 'param':
            param_json = json.dumps(msg_value).encode('utf-8')
            writer.write(struct.pack('!hI', w.MessageReceive.SET_PARAM, len(param_json)) + param_json)

        await writer.drain()


async def worker_receiver(
        ws: WebSocket,
        reader: asyncio.StreamReader,
):
    while True:
        msg_type_data = await reader.readexactly(2)
        msg_type, = struct.unpack('!h', msg_type_data)
        print(f'worker -> API: received {w.MessageSend(msg_type).name}')

        match msg_type:

            case w.MessageSend.PARAMS:
                length_data = await reader.readexactly(4)
                length, = struct.unpack('!I', length_data)
                params_data = await reader.readexactly(length)
                params = json.loads(params_data.decode('utf-8'))
                await ws.send_json({'params': params})

            case w.MessageSend.LOSS:
                data = await reader.readexactly(4)
                loss, = struct.unpack('!f', data)
                await ws.send_json({'loss': loss})

            case w.MessageSend.TEST_PATH:
                metadata = await reader.readexactly(4)
                inputs, num_steps, = struct.unpack('!2h', metadata)

                data = await reader.readexactly(4 * inputs * num_steps)
                test_path = struct.unpack(f'!{int(inputs * num_steps)}f', data)
                t = test_path[:num_steps]
                s = test_path[num_steps:2*num_steps]
                d = test_path[2*num_steps:]
                await ws.send_json({'test_path': {
                    't': t,
                    's': s,
                    'd': d,
                }})


async def worker_messaging(
        ws: WebSocket,
        message_queue: asyncio.Queue[tuple[str, str]],
        socket_path: str
):
    while True:
        try:
            reader, writer = await asyncio.open_unix_connection(socket_path)
            print('Connected to worker')

            sender_task = asyncio.create_task(worker_sender(message_queue, writer))
            receiver_task = asyncio.create_task(worker_receiver(ws, reader))

            done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()

        except (ConnectionRefusedError, FileNotFoundError):
            print('Worker not ready, retrying...')
            await asyncio.sleep(0.5)

        except asyncio.IncompleteReadError:
            print('Worker disconnected, reconnecting...')
            await asyncio.sleep(0.5)

        except Exception as e:
            print(e)


def configure_ws(app: FastAPI):

    @app.websocket('/ws')
    async def _(ws: WebSocket):

        await ws.accept()
        message_queue = asyncio.Queue()

        socket_path = f'/tmp/worker-{next(counter)}.sock'

        worker_process = subprocess.Popen(
            ['python', 'worker.py', socket_path],
        )

        listener_task = asyncio.create_task(worker_messaging(ws, message_queue, socket_path))

        try:
            while True:
                parameters = await ws.receive_json()
                print(f'client -> API: received {parameters}')

                if 'event' in parameters:
                    await message_queue.put(('event', parameters['event']))

                for key, value in parameters.items():
                    if key != 'event':
                        await message_queue.put(('param', {key: value}))

        except WebSocketDisconnect:
            print('disconnected')
            listener_task.cancel()
            worker_process.terminate()
        except Exception as e:
            print(e)


def configure_static(app: FastAPI):
    plotly_dir = os.path.join(os.path.dirname(plotly.__file__), 'package_data')
    app.mount('/plotly', StaticFiles(directory=plotly_dir), name='plotly')

    static_dir = os.path.join(os.path.dirname(__file__), STATIC_DIR_NAME)
    static_files = StaticFiles(directory=static_dir)

    @app.get('/')
    async def root():
        return RedirectResponse('/index.html')

    app.mount('/', static_files)


LOGIN_HTML = '''<!DOCTYPE html>
<html><head><title>cher - login</title>
<style>body{background:#1e1e2e;color:#cdd6f4;font-family:monospace;display:flex;justify-content:center;align-items:center;height:100vh;margin:0}
form{display:flex;flex-direction:column;gap:8px}input{padding:6px;font-family:monospace}</style>
</head><body><form method="post" action="/login">
<input type="password" name="password" placeholder="password" autofocus>
<input type="submit" value="login">
</form></body></html>'''


class CookieAuthMiddleware:
    def __init__(self, app, password: str):
        self.app = app
        self.password = password

    def _get_cookie(self, headers: list[tuple[bytes, bytes]]) -> str:
        for key, value in headers:
            if key == b'cookie':
                for item in value.decode().split(';'):
                    item = item.strip()
                    if item.startswith('cher_token='):
                        return item.split('=', 1)[1]
        return ''

    async def __call__(self, scope, receive, send):
        if scope['type'] not in ('http', 'websocket'):
            await self.app(scope, receive, send)
            return

        path = scope.get('path', '')

        if path == '/login':
            await self.app(scope, receive, send)
            return

        token = self._get_cookie(scope.get('headers', []))
        if secrets.compare_digest(token, self.password):
            await self.app(scope, receive, send)
            return

        if scope['type'] == 'websocket':
            await send({'type': 'websocket.close', 'code': 1008})
            return

        response = RedirectResponse('/login')
        await response(scope, receive, send)


def configure_auth(app: FastAPI, password: str):

    @app.get('/login', response_class=HTMLResponse)
    async def login_page():
        return LOGIN_HTML

    @app.post('/login')
    async def login(request: Request):
        form = await request.form()
        pw = form.get('password', '')
        if secrets.compare_digest(pw, password):
            response = RedirectResponse('/', status_code=303)
            response.set_cookie('cher_token', pw, httponly=True)
            return response
        return RedirectResponse('/login', status_code=303)


def main():
    app = FastAPI()

    password = os.environ.get('CHER_PASSWORD')

    configure_ws(app)

    if password:
        configure_auth(app, password)
        app.add_middleware(CookieAuthMiddleware, password=password)

    configure_static(app)

    uvicorn.run(app, host='0.0.0.0', port=9000)


if __name__ == '__main__':
    main()
