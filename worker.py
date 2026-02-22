import os
import sys
import json
import time
import select
import socket
import struct
import itertools
import mojo.importer
from enum import IntEnum
from cher_mojo import Network

BATCH_UPDATE_SIZE = 10

class MessageReceive(IntEnum):
    START              = 1
    STOP               = 2
    GENERATE_TEST_PATH = 3
    SET_PARAM          = 4

class MessageSend(IntEnum):
    LOSS      = 1
    TEST_PATH = 2
    PARAMS    = 3


def recv_exactly(conn, n):
    data = bytearray()
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError('client disconnected')
        data.extend(chunk)
    return bytes(data)


def send_test_path(conn, test_path):
    inputs    = test_path.get_inputs()
    num_steps = test_path.get_num_steps()
    data      = test_path.get_data()

    msg = struct.pack(
            f'!3h{int(inputs * num_steps)}f',
            MessageSend.TEST_PATH,
            inputs,
            num_steps,
            *list(itertools.chain.from_iterable(data)),
    )
    conn.sendall(msg)


def main():

    iteration = 0
    running = False
    step_time_sum = 0.0

    socket_path = sys.argv[1]

    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)

    network = Network()
    test_path = network.test_path()

    while True:
        conn, _ = server.accept()
        print('Client connected')

        try:
            params = network.get_params()
            for key, value in params.items():
                if isinstance(value, float):
                    params[key] = float(f'{value:.6g}')
            params_json = json.dumps(params).encode('utf-8')
            msg = struct.pack('!hI', MessageSend.PARAMS, len(params_json)) + params_json
            conn.sendall(msg)

            while True:
                timeout = 0 if running else 0.05
                readable, _, _ = select.select([conn], [], [], timeout)

                if readable:
                    msg_type_data = recv_exactly(conn, 2)
                    msg_type, = struct.unpack('!h', msg_type_data)
                    print(f'worker received msg type: {MessageReceive(msg_type).name}')

                    match msg_type:
                        case MessageReceive.START:
                            running = True
                        case MessageReceive.STOP:
                            running = False
                        case MessageReceive.GENERATE_TEST_PATH:
                            print('Generate test path')
                            test_path = network.test_path()
                            network.run_test(test_path)
                            send_test_path(conn, test_path)
                        case MessageReceive.SET_PARAM:
                            length_data = recv_exactly(conn, 4)
                            length, = struct.unpack('!I', length_data)
                            param_data = recv_exactly(conn, length)
                            param = json.loads(param_data.decode('utf-8'))
                            name, value = next(iter(param.items()))
                            print(f'Set {name}: {value}')
                            match name:
                                case 'slippage':
                                    network.set_slippage(value)
                                    network.reset_counters()
                                case 'strike':
                                    network.set_strike(value)
                                    network.reset_counters()

                if running:
                    try:
                        t0 = time.perf_counter()
                        network.run()
                        step_time = time.perf_counter() - t0

                        iteration += 1
                        step_time_sum += step_time
                        loss = network.loss()

                        if iteration % BATCH_UPDATE_SIZE == 0:
                            if test_path:
                                network.run_test(test_path)

                            avg_step_time = step_time_sum / BATCH_UPDATE_SIZE
                            step_time_sum = 0.0

                            msg = struct.pack('!hff', MessageSend.LOSS, loss, avg_step_time)
                            conn.sendall(msg)

                            send_test_path(conn, test_path)
                    except Exception as e:
                        print(e)

        except ConnectionError:
            print('Client disconnected')
        except Exception as e:
            print(f'Error: {e}')
        finally:
            conn.close()

if __name__ == '__main__':
    main()
