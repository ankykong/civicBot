import argparse
import json
from argparse import RawTextHelpFormatter
import requests
from typing import Optional, Dict
import warnings
try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn(
        "Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None

BASE_API_URL = "http://127.0.0.1:7861"
FLOW_ID = "d8fcb38d-6b7f-495a-af5f-f9f053c866f5"
ENDPOINT = ""  # You can set a specific endpoint name in the flow settings

# You can tweak the flow by adding a tweaks dictionary
# e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
TWEAKS = {
    "Prompt-dmbgN": {},
    "ChatInput-Yegqn": {},
    "Prompt-Iflb8": {},
    "Agent-hT3OW": {},
    "Prompt-8zoSD": {},
    "Prompt-ScgrD": {},
    "TavilySearchComponent-UlplS": {},
    "OpenAIModel-ZlMnb": {},
    "OpenAIModel-38fKB": {},
    "ChatOutput-wCMZD": {},
    "Prompt-T9dPI": {},
    "OpenAIModel-2V6s0": {},
    "OpenAIModel-GYGHT": {},
    "Prompt-dqXja": {}
}


class LangflowLLM:
    def __init__(
        self,
        base_api_url: str = "http://127.0.0.1:7861",
        flow_id: str = "d8fcb38d-6b7f-495a-af5f-f9f053c866f5",
        endpoint: str = "",  # You can set a specific endpoint name in the flow settings
        tweaks: Optional[Dict[str, Dict]] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LangflowLLM client.

        :param base_api_url: The base URL for the Langflow API
        :param flow_id: The ID of the flow to use
        :param endpoint: The endpoint name of the flow (if specified)
        :param tweaks: Optional tweaks to customize the flow
        :param api_key: Optional API key for authentication
        """
        self.base_api_url = base_api_url
        self.flow_id = flow_id
        self.endpoint = endpoint or flow_id
        self.api_key = api_key

        # Default tweaks if none provided
        self.tweaks = tweaks or {
            "Prompt-dmbgN": {},
            "ChatInput-Yegqn": {},
            "Prompt-Iflb8": {},
            "Agent-hT3OW": {},
            "Prompt-8zoSD": {},
            "Prompt-ScgrD": {},
            "TavilySearchComponent-UlplS": {},
            "OpenAIModel-ZlMnb": {},
            "OpenAIModel-38fKB": {},
            "ChatOutput-wCMZD": {},
            "Prompt-T9dPI": {},
            "OpenAIModel-2V6s0": {},
            "OpenAIModel-GYGHT": {},
            "Prompt-dqXja": {}
        }

    def run_flow(self,
                 message: str,
                 output_type: str = "chat",
                 input_type: str = "chat",) -> dict:
        """
        Run a flow with a given message and optional tweaks.

        :param message: The message to send to the flow
        :param endpoint: The ID or the endpoint name of the flow
        :param tweaks: Optional tweaks to customize the flow
        :return: The JSON response from the flow
        """
        api_url = f"{self.base_api_url}/api/v1/run/{self.endpoint}"
        print(api_url)

        payload = {
            "input_value": message,
            "output_type": output_type,
            "input_type": input_type,
        }
        headers = None
        if self.tweaks:
            payload["tweaks"] = self.tweaks
        if self.api_key:
            headers = {"x-api-key": self.api_key}

        print(payload)
        print(headers)
        response = requests.post(api_url, json=payload, headers=headers)
        if response and hasattr(response, 'json'):
            response = response.json()
            print(json.dumps(response))
            return response.get(
                "outputs",
                {})[-1].get(
                "outputs",
                {})[-1].get(
                    "results",
                    {}).get(
                        "message",
                        {}).get(
                            "data",
                {}).get("text")
        else:
            return ""


def run_flow(message: str,
             endpoint: str,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"
    print(api_url)
    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers = {"x-api-key": api_key}

    print(payload)
    print(headers)
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="""Run a flow with a given message and optional tweaks.
Run it like: python <your file>.py "your message here" --endpoint "your_endpoint" --tweaks '{"key": "value"}'""",
        formatter_class=RawTextHelpFormatter)
    parser.add_argument("message", type=str, help="The message to send to the flow")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT or FLOW_ID,
                        help="The ID or the endpoint name of the flow")
    parser.add_argument(
        "--tweaks",
        type=str,
        help="JSON string representing the tweaks to customize the flow",
        default=json.dumps(TWEAKS))
    parser.add_argument("--api_key", type=str, help="API key for authentication", default=None)
    parser.add_argument("--output_type", type=str, default="chat", help="The output type")
    parser.add_argument("--input_type", type=str, default="chat", help="The input type")
    parser.add_argument("--upload_file", type=str, help="Path to the file to upload", default=None)
    parser.add_argument(
        "--components",
        type=str,
        help="Components to upload the file to",
        default=None)

    args = parser.parse_args()
    try:
        tweaks = json.loads(args.tweaks)
    except json.JSONDecodeError:
        raise ValueError("Invalid tweaks JSON string")

    if args.upload_file:
        if not upload_file:
            raise ImportError(
                "Langflow is not installed. Please install it to use the upload_file function.")
        elif not args.components:
            raise ValueError("You need to provide the components to upload the file to.")
        tweaks = upload_file(
            file_path=args.upload_file,
            host=BASE_API_URL,
            flow_id=args.endpoint,
            components=[
                args.components],
            tweaks=tweaks)

    llm = LangflowLLM()

    response = llm.run_flow(
        message=args.message,
        output_type=args.output_type,
        input_type=args.input_type,
    )

    print(
        response.get(
            "outputs",
            {})[0].get(
            "outputs",
            {})[0].get(
                "results",
                {}).get(
                    "message",
                    {}).get(
                        "data",
            {}).get("text"))


if __name__ == "__main__":
    main()
