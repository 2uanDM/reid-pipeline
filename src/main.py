import os
import sys

sys.path.append(os.getcwd())
from src.server.main import MainServer

server = MainServer()
server.run()
