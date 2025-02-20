import logging
import os
from dotenv import load_dotenv

load_dotenv()

public_or_local = os.getenv("PUBLIC_OR_LOCAL", "LOCAL")


logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.ERROR,
)
logger = logging.getLogger("microservice indicators")
