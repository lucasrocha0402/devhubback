import logging
import traceback
from main import app  

logging.getLogger('werkzeug').setLevel(logging.ERROR)

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5002)
    except Exception as e:
        print(f"Erro ao iniciar servidor: {e}")
        traceback.print_exc()
