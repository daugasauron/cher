import os
import json
import time
import worker as w
import struct
import plotly
import asyncio
import uvicorn
import subprocess
import mojo.importer
from itertools import count
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from concurrent.futures import ProcessPoolExecutor
from cher_mojo import Network

STATIC_DIR_NAME = 'static'

counter = count(0)

async def worker_messaging(
        ws: WebSocket, 
        message_queue: asyncio.Queue[str], 
        socket_path: str
):
    while True:
        try:
            reader, writer = await asyncio.open_unix_connection(socket_path)
            print('Connected to worker')

            while True:
                try:
                    msg_type, msg_value = message_queue.get_nowait()
                    print(f'{msg_type}: {msg_value}')

                    if msg_type == 'event' and msg_value == 'generate_path':
                        msg = struct.pack('!h', w.MessageReceive.GENERATE_TEST_PATH)
                        writer.write(msg)

                    elif msg_type == 'slippage':
                        msg = struct.pack('!hf', w.MessageReceive.SET_SLIPPAGE, msg_value)
                        writer.write(msg)

                except asyncio.QueueEmpty:
                    pass

                msg_type_data = await reader.readexactly(2)
                msg_type, = struct.unpack('!h', msg_type_data)

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
                print(f'received JSON: {parameters}')

                if 'event' in parameters:
                    message_queue.put_nowait(('event', parameters['event']))

                if 'slippage' in parameters:
                    message_queue.put_nowait(('slippage', parameters['slippage']))

        except WebSocketDisconnect:
            print('disconnected')
            listener_task.cancel()
            worker_process.terminate()


def configure_static(app: FastAPI):
    plotly_dir = os.path.join(os.path.dirname(plotly.__file__), 'package_data')
    app.mount('/plotly', StaticFiles(directory=plotly_dir), name='plotly')

    static_dir = os.path.join(os.path.dirname(__file__), STATIC_DIR_NAME)
    static_files = StaticFiles(directory=static_dir)
    app.mount('/', static_files)


def main():
    app = FastAPI()

    configure_ws(app)
    configure_static(app)

    uvicorn.run(app)


if __name__ == '__main__':
    main()
