import uvicorn
from api import app, stream_video
import threading

if __name__ == '__main__':
    stream_vid = threading.Thread(
            target=stream_video,
            daemon=True,
            )
    stream_vid.start()
    uvicorn.run(app, host='localhost', port=8000)
    stream_vid.join()
