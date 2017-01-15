import json
import asyncio
import threading
from concurrent.futures import CancelledError

import numpy as np

loop = asyncio.get_event_loop()

async def _handler(name, shape, get_bitmap, FPS, host, port):
    transport = None
    fail_cnt = 0
    while not transport and fail_cnt < 10:
        try:
            transport, _ = await loop.create_connection(asyncio.Protocol,
                                                        host=host,
                                                        port=port)
        except OSError:
            print('failed to connect the env viewer, retry...')
            fail_cnt += 1
            await asyncio.sleep(1)
    if not transport:
        print('give up sending observation to the env viewer. :(')
        return
    transport.writelines([name.encode()])
    await asyncio.sleep(0.01)
    transport.writelines([json.dumps(shape).encode()])
    await asyncio.sleep(0.01)
    while not transport.is_closing():
        try:
            data = get_bitmap().astype(np.uint8)
            transport.writelines([data.tobytes()])
            await asyncio.sleep(1 / FPS)
        except (CancelledError, KeyboardInterrupt):
            break

    if not transport.is_closing():
        transport.write_eof()

def run(name, shape, get_bitmap, FPS, host, port):
    try:
        loop.run_until_complete(_handler(name, shape, get_bitmap,
                                         FPS, host, port))
        loop.run_forever()
    except KeyboardInterrupt:
        loop.stop()
    except RuntimeError:
        pass

def start_send_thread(name, shape, get_bitmap, FPS,
                      host='localhost', port=12345):
    global loop
    loop = asyncio.get_event_loop()
    th = threading.Thread(target=run, args=(name, shape, get_bitmap, FPS,
                                            host, port))
    th.start()
    def stop():
        loop.stop()
        th.join()
    return stop
