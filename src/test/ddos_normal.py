import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from src.utils.feature_extraction import FeatureExtractor

executor = FeatureExtractor()


def make_request(image_path, request_id):
    try:
        # Load image
        image = cv2.imread(image_path)

        # Extract features
        start_time = time.time()
        features = executor.inference(image)
        end_time = time.time()

        print(f"Request {request_id} completed in {end_time - start_time:.2f}s")
        return end_time - start_time
    except Exception as e:
        print(f"Request {request_id} failed: {str(e)}")
        return None


def load_test_multithread(num_requests, concurrency, image_path):
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all requests to the thread pool
        futures = [
            executor.submit(make_request, image_path, i) for i in range(num_requests)
        ]

        # Wait for all requests to complete
        durations = []
        for future in futures:
            duration = future.result()
            if duration is not None:
                durations.append(duration)

    return durations


def load_test_for_loop(num_requests, image_path):
    durations = []
    for i in range(num_requests):
        duration = make_request(image_path, i)
        if duration is not None:
            durations.append(duration)

    return durations


if __name__ == "__main__":
    NUM_REQUESTS = 1000  # Total number of requests to make
    CONCURRENCY = 20  # Number of concurrent requests
    IMAGE_PATH = "screenshot-20250309-233405.png"

    # start_time = time.time()

    # durations = load_test_multithread(NUM_REQUESTS, CONCURRENCY, IMAGE_PATH)

    # end_time = time.time()
    # total_time = end_time - start_time

    # print("\nLoad Test Results:")
    # print(f"Total Requests: {NUM_REQUESTS}")
    # print(f"Concurrency Level: {CONCURRENCY}")
    # print(f"Total Time: {total_time:.2f}s")
    # print(f"Average RPS: {NUM_REQUESTS / total_time:.2f}")
    # print(f"Average Response Time: {sum(durations) / len(durations):.2f}s")
    # print(f"Min Response Time: {min(durations):.2f}s")
    # print(f"Max Response Time: {max(durations):.2f}s")

    start_time = time.time()

    durations = load_test_for_loop(NUM_REQUESTS, IMAGE_PATH)

    end_time = time.time()
    total_time = end_time - start_time

    print("\nLoad Test Results:")
    print(f"Total Requests: {NUM_REQUESTS}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average RPS: {NUM_REQUESTS / total_time:.2f}")
    print(f"Average Response Time: {sum(durations) / len(durations):.2f}s")
    print(f"Min Response Time: {min(durations):.2f}s")
    print(f"Max Response Time: {max(durations):.2f}s")
