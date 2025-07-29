import json
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment
from tokenizers import Encoding
from tokenizers import Tokenizer as TokenizerBase


class Tokenizer:
    """
    Tokenizer class that supports chat templates using the Jinja2 engine.

    This class integrates custom tokenizer configurations and templates to encode chat messages into a format acceptable by the model.
    """

    def __init__(self, tokenizer_path: str):
        """
        Initializes the Tokenizer instance.

        Args:
            tokenizer_path (str): The path to the tokenizer model file.
        """
        super().__init__()
        # Build the path to tokenizer_config.json, assuming it is in the same directory as tokenizer_path
        tokennizer_config_path = Path(tokenizer_path).parent / "tokenizer_config.json"
        self.tokenizer_config = json.load(open(tokennizer_config_path))

        # Initialize the tokenizer, loading the tokenizer model using the specified tokenizer_path
        self.tokenizer = TokenizerBase.from_file(tokenizer_path)

        # Use the Jinja2 environment to load and parse the chat template string
        self.chat_template = Environment().from_string(self.tokenizer_config["chat_template"])

        self.eos_token = self.tokenizer_config["eos_token"]
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)

        # Get the PAD (padding) token and its corresponding ID
        self.pad_token = self.tokenizer_config["pad_token"]
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)

    def encode_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Encodes a list of chat messages into a string format using a predefined chat template.

        Args:
            messages (List[Dict[str, str]]): A list of chat messages, where each message is a dictionary
                containing 'role' and 'content' keys.
                For example:
                [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hello! How can I help you?"}
                ]

        Returns:
            str: The encoded string, containing all messages and a generation prompt (if enabled).
        """
        # Render the messages using the Jinja2 template, adding a generation prompt (if enabled in the configuration)
        return self.chat_template.render(messages=messages, add_generation_prompt=True)

    def encode_chat_with_response_prompt(
        self, messages: List[Dict[str, str]], prompt: str
    ) -> str:
        """
        Encodes a list of chat messages into a string and appends a response prompt.

        Args:
            messages (List[Dict[str, str]]): A list of chat messages, in the same format as above.
            prompt (str): The response prompt string to append.

        Returns:
            str: The encoded string, containing all messages and the appended response prompt.
        """
        return self.encode_chat(messages) + prompt

    def tokenize(self, text: str) -> Encoding:
        """
        Tokenizes a text string and converts it to token IDs.

        Args:
            text (str): The text to be tokenized.

        Returns:
            Encoding: The tokenization result, containing token IDs and other relevant information.
        """
     
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        """
        Converts a list of token IDs back into a text string.

        Args:
            token_ids (List[int]): The list of token IDs to be converted back.

        Returns:
            str: The converted text string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)