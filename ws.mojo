from python import Python, PythonObject

fn main() raises:
    app = Python.import_module('app')
    time = Python.import_module('time')

    state = Python.dict()
    server = app.create_app(state)

    state['iter'] = 0

    while not server.should_exit:
        time.sleep(0.1)
        state['iter'] += 1
