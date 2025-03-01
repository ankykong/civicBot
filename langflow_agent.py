# import asyncio
import os
from pathlib import Path
import logging
from typing import Optional, Union

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from langflow_llm import LangflowLLM


logger = logging.getLogger("jinn-pipecat")


class LangFlowAgent(FrameProcessor):
    """
    Agent as a Pipecat service.
    """

    def __init__(self, config_path: Optional[Union[str, Path]]
                 = None, config: Optional[dict] = None):
        super().__init__()

        self.agent = LangflowLLM()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Processes the incoming frames if relevant.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMMessagesFrame):

            # Sometimes messages come from the end of the pipeline - bounce them back
            # downstream towards TTS
            direction = FrameDirection.DOWNSTREAM
            await self.push_frame(LLMFullResponseStartFrame(), direction)

            user_message = frame.messages[-1]["content"] if frame.messages else None

            print(user_message)

            full_response = self.agent.run_flow(user_message)
            await self.push_frame(TextFrame(full_response), direction)
            await self.push_frame(LLMFullResponseEndFrame(), direction)

        else:
            await self.push_frame(frame, direction)
