import asyncio
import time
from pathlib import Path

import aiohttp


async def make_request(session, url, image_path, request_id):
    try:
        data = aiohttp.FormData()
        data.add_field(
            "image",
            open(image_path, "rb"),
            filename=Path(image_path).name,
            content_type="image/png",
        )

        start_time = time.time()
        async with session.post(url, data=data) as response:
            await response.json()
            end_time = time.time()
            print(f"Request {request_id} completed in {end_time - start_time:.2f}s")
            return end_time - start_time
    except Exception as e:
        print(f"Request {request_id} failed: {str(e)}")
        return None


async def load_test(num_requests, concurrency, image_path):
    url = "http://localhost:8000/embedding"

    async with aiohttp.ClientSession() as session:
        tasks = set()  # Change to set instead of list
        for i in range(num_requests):
            if len(tasks) >= concurrency:
                # Wait for some tasks to complete before adding more
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                # Update tasks with pending tasks
                tasks = pending
                # Process completed tasks
                for task in done:
                    await task

            task = asyncio.create_task(make_request(session, url, image_path, i))
            tasks.add(task)  # Use add() instead of append()

        # Wait for remaining tasks
        if tasks:
            done, _ = await asyncio.wait(tasks)
            for task in done:
                await task


if __name__ == "__main__":
    NUM_REQUESTS = 10000  # Total number of requests to make
    CONCURRENCY = 100  # Number of concurrent requests
    IMAGE_PATH = "screenshot-20250309-233405.png"  # Path to your test image

    start_time = time.time()

    asyncio.run(load_test(NUM_REQUESTS, CONCURRENCY, IMAGE_PATH))

    end_time = time.time()
    total_time = end_time - start_time

    print("\nLoad Test Results:")
    print(f"Total Requests: {NUM_REQUESTS}")
    print(f"Concurrency Level: {CONCURRENCY}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average RPS: {NUM_REQUESTS / total_time:.2f}")
