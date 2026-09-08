"""Start the local gateway UI. Colab remains the GPU workplace."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


def main() -> None:
    print("Gateway UI: http://127.0.0.1:8765")
    print("Queue jobs here. Run workplace/colab_worker.ipynb on Colab A100.")
    uvicorn.run(
        "gateway.app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
