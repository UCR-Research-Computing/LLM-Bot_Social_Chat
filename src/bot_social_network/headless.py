import asyncio
import argparse
import logging
import time
import os
from typing import Optional

from .simulation import Simulation
from .database import close_database_connection
from . import logging_config
from . import voice_manager


async def main():
    parser = argparse.ArgumentParser(
        prog="Bot Social Network (Headless Experiment Runner)",
        description="""
Runs the bot simulation without the TUI, designed for experiments and automated runs.
All output is directed to a unique, timestamped, structured log file in the 'logs/' directory.

This script is ideal for running controlled experiments where the primary output is the
log file for later analysis. For interactive viewing, see main.py.
""",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
--------------------------------------------------------------------------------
Usage Examples:
  # Run a short experiment with a specific config and topic
  python3 headless.py --config example_gemma3n.json --max-posts 20 --topic "What is the nature of memory?"

  # Run a 5-minute deterministic simulation with TTS enabled (audio will play)
  python3 headless.py --config gemini_models_showcase.json --duration 300 --deterministic --tts --autostart
--------------------------------------------------------------------------------
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default.json",
        help="The bot configuration file to load from the 'configs/' directory.",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable text-to-speech (will slow down the simulation).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Maximum duration of the simulation in seconds. The simulation will stop after this time.",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=None,
        help="Maximum number of posts to generate before stopping.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="An initial topic to inject into the conversation to guide the simulation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Select bots in a predictable, round-robin order instead of randomly.",
    )
    args = parser.parse_args()

    if not args.duration and not args.max_posts:
        parser.error(
            "Headless mode requires either --duration or --max-posts to be set."
        )

    run_dir = logging_config.setup_logging()
    audio_dir = os.path.join(run_dir, "audio")
    print(f"Logging simulation to: {run_dir}")

    sim = Simulation(
        config_file=args.config,
        autostart=True,  # Always autostart in headless
        tts_enabled=args.tts,
        clear_db=True,  # Always start fresh in headless
        max_posts=args.max_posts,
        duration=args.duration,
        topic=args.topic,
        deterministic=args.deterministic,
        audio_dir=audio_dir,
    )

    await sim.initialize()

    # Start the workers if TTS is enabled
    generation_task = None
    speaker_task = None
    if sim.tts_enabled:
        generation_queue = asyncio.Queue()
        playback_queue = asyncio.Queue(maxsize=1)  # Ensures one-at-a-time playback

        generation_task = asyncio.create_task(
            generation_worker(sim, generation_queue, playback_queue)
        )
        speaker_task = asyncio.create_task(speaker_worker(sim, playback_queue))

    # Start the simulation loop
    simulation_task = asyncio.create_task(
        run_simulation(sim, generation_queue if sim.tts_enabled else None)
    )
    try:
        await simulation_task
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        logging.info(
            "Simulation interrupted by user.", extra={"event": "sim.end.interrupt"}
        )
        simulation_task.cancel()
    finally:
        print("Shutting down...")
        # Gracefully handle shutdown
        if generation_task:
            await generation_queue.join()
            generation_task.cancel()
        if speaker_task:
            speaker_task.cancel()

        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        close_database_connection()
        print("Shutdown complete.")


async def run_simulation(sim: Simulation, generation_queue: Optional[asyncio.Queue]):
    """The main simulation loop."""
    start_time = time.time()
    while True:
        if sim.duration and (time.time() - start_time) > sim.duration:
            logging.info(
                "Simulation ended: duration reached.",
                extra={"event": "sim.end.duration"},
            )
            break
        if sim.max_posts and sim.post_count >= sim.max_posts:
            logging.info(
                "Simulation ended: max posts reached.",
                extra={"event": "sim.end.max_posts"},
            )
            break

        post = await sim.run_bot_activity()
        if post:
            if sim.tts_enabled and generation_queue:
                await generation_queue.put(post)
            else:
                print(f"{post.sender}: {post.content}")

        if not sim.tts_enabled:
            await asyncio.sleep(1)


async def generation_worker(
    sim: Simulation, in_queue: asyncio.Queue, out_queue: asyncio.Queue
):
    """The worker that generates audio files from posts."""
    while True:
        try:
            post = await in_queue.get()
            if not post.sender == "SYSTEM":
                voice_name = voice_manager.select_voice(post.sender)
                if voice_name and sim.audio_dir:
                    output_path = os.path.join(sim.audio_dir, f"post_{post.id}.mp3")
                    success = await voice_manager.generate_voice_file(
                        post.content, voice_name, output_path
                    )
                    if success:
                        await out_queue.put({"post": post, "audio_path": output_path})
            in_queue.task_done()
        except asyncio.CancelledError:
            break


async def speaker_worker(sim: Simulation, in_queue: asyncio.Queue):
    """The consumer task that plays audio and prints posts from the queue."""
    while True:
        try:
            data = await in_queue.get()
            post = data["post"]
            audio_path = data["audio_path"]

            print(f"{post.sender}: {post.content}")
            await asyncio.to_thread(voice_manager.play_audio_file, audio_path)

            in_queue.task_done()
        except asyncio.CancelledError:
            break


if __name__ == "__main__":
    asyncio.run(main())
