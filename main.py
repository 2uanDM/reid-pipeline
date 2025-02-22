import argparse

from src.edge.base import BaseEdgeDevice
from src.server.main import MainServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run edge device
    parser.add_argument("--edge", action="store_true", help="Run in edge mode")
    parser.add_argument("--cpu", action="store_true", help="Use CPU device")
    parser.add_argument("--jetson", action="store_true", help="Use Jetson device")
    parser.add_argument("--source", type=str, help="Source path")

    # Run server device
    parser.add_argument("--server", action="store_true", help="Run in server mode")
    args = parser.parse_args()

    if args.edge and args.server:
        raise ValueError("Only run edge or server device")
    elif not args.edge and not args.server:
        raise ValueError("Please specify edge or server device")
    elif args.edge:
        if args.edge and args.jetson:
            raise ValueError("Choose only one edge device type")
        elif args.cpu:
            first_device = BaseEdgeDevice(
                source=args.source,
            )
            first_device.run()
        elif args.jetson:
            raise NotImplementedError("Jetson device not implemented")
        else:
            raise ValueError("Please specify edge device type")
    elif args.server:
        server = MainServer()
        server.run()
