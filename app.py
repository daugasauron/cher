from threading import Thread
from uvicorn import Config, Server
from fastapi import FastAPI


def create_app(state) -> Server:
    app = FastAPI()

    @app.get('/state')
    def get_state():
        return state

    config = Config(
        app,
        host='127.0.0.1',
        port=8000,
        log_level='info',
    )

    server = Server(config)

    def run():
        print('uvicorn thread starting')
        server.run()
        print('uvicorn thread exited')

    thread = Thread(target=run, daemon=True)
    thread.start()

    return server
