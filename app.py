import os

from waitress import serve

from endgrain.web import create_app

app = create_app()

if __name__ == "__main__":
    serve(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "18948")),
        threads=int(os.environ.get("ENDGRAIN_THREADS", "4")),
    )
