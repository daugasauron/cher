import os
import sys
import json
import socket
import struct
import itertools
import mojo.importer
from cher_mojo import Network

BATCH_UPDATE_SIZE = 10

class MessageReceive:
    GENERATE_TEST_PATH: int = 1
    SET_SLIPPAGE: int       = 2

class MessageSend:
    LOSS: int      = 1
    TEST_PATH: int = 2
    PARAMS: int    = 3


def main():

    iteration = 0

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
        conn.setblocking(False)

        print('Client connected')

        # Send initial params to client
        params = network.get_params()
        # Clean up float precision issues using significant figures
        for key, value in params.items():
            if isinstance(value, float):
                params[key] = float(f'{value:.6g}')
        params_json = json.dumps(params).encode('utf-8')
        msg = struct.pack('!hI', MessageSend.PARAMS, len(params_json)) + params_json
        conn.sendall(msg)

        while True:

            try:
                msg_type_data = conn.recv(2)
                msg_type, = struct.unpack('!h', msg_type_data)
                print(f'worker received msg type: {msg_type}')

                match msg_type:
                    case MessageReceive.GENERATE_TEST_PATH:
                        print('Generate test path')
                        test_path = network.test_path()

                    case MessageReceive.SET_SLIPPAGE:
                        data = conn.recv(4)
                        slippage, = struct.unpack('!f', data)
                        print(f'Set slippage: {slippage}')
                        network.set_slippage(slippage)
                        network.reset_counters()


            except BlockingIOError as e:
                pass
            except Exception as e:
                print(e)

            network.run()
            iteration += 1

            loss = network.loss()

            if iteration % BATCH_UPDATE_SIZE == 0:
                print(f'loss: {loss}')
                if test_path:
                    network.run_test(test_path)

                msg = struct.pack('!hf', MessageSend.LOSS, loss)
                conn.sendall(msg)

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

if __name__ == '__main__':
    main()
