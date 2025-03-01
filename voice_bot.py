import asyncio
import aiohttp
import os
import sys
import json

from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.ollama import OLLamaLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator
)

from runner import configure
from loguru import logger
from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def start_interview_bot():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Civic Bot",
            DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                camera_out_enabled=False,
                transcription_enabled=False,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        stop_secs=0.2,
                        start_secs=0.2,
                        confidence=0.4,
                    )
                )
            )
        )

        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            language="en-US",
            model="nova"
        )

        tts = DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY")
        )

        llm = OLLamaLLMService(model="llama3.2")

        messages = []

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                tma_in,
                llm,
                tts,
                transport.output(),
                tma_out,
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            )
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            participant_name = participant.get("info", {}).get("userName", "")
            system_prompt = SYSTEM_PROMPT.format(name=participant_name)
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": "Hello"})
            await task.queue_frames([LLMMessagesFrame(messages)])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)
        return json.dumps({
            "status": 200,
            "score": 1.0,
        })

if __name__ == "__main__":
    asyncio.run(start_interview_bot())
