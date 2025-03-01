import argparse
import os
from typing import Optional

import aiohttp

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper


async def configure(aiohttp_session: aiohttp.ClientSession):
    (url, token, _) = await configure_with_args(aiohttp_session)
    return (url, token)


async def configure_with_args(
        aiohttp_session: aiohttp.ClientSession,
        parser: Optional[argparse.ArgumentParser] = None):
    if not parser:
        parser = argparse.ArgumentParser(description="Daily AI SDK Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="URL of the Daily Room to Join"
    )
    parser.add_argument("-k", "--apikey", type=str, required=False,
                        help="Daily API Key (needed to create an owner token for the room)")

    args, unknown = parser.parse_known_args()
    url = args.url or os.getenv("DAILY_SAMPLE_ROOM_URL")
    key = args.apikey or os.getenv("DAILY_API_KEY")

    if not url:
        raise Exception(
            "No Daily room specified. Use the -u/--url option from the command line or set DAILY_SAMPLE_ROOM_URL in your environment to specify a Daily room URL."
        )

    if not key:
        raise Exception(
            "No Daily API Key specifice. Use the -k/--apikey option from the command line or set DAILY_API_KEY in your environment to specify a Daily API key available from the https://dashboard.daily.co/develpers."
        )

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    expiry_time: float = 5 * 60

    token = await daily_rest_helper.get_token(url, expiry_time)

    return (url, token, args)
