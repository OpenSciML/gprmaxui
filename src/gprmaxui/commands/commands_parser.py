from __future__ import annotations

import logging
import re
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Any

from pydantic import BaseModel, create_model, Field

logger = logging.getLogger("rich")


def patch_model(model: BaseModel, **fields: Dict[str, Any]) -> BaseModel:
    """
    Dynamically create a new Pydantic model by extending an existing model with additional fields.

    :param model: The base model to extend.
    :param fields: Key-value pairs of fields to add to the model.
    :return: A new Pydantic model with the additional fields.
    """
    new_model = create_model(
        model.__name__,
        **fields,
        __base__=model
    )
    return new_model


class BaseCommand(BaseModel):
    """
    Abstract class representing a command.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """
        Execute the command and print it.

        :param args: Positional arguments for command execution.
        :param kwargs: Keyword arguments for command execution.
        """
        self.print()

    def print(self) -> None:
        """
        Print the string representation of the command.
        """
        print(self)


class Command(BaseCommand):
    """
    Base class for individual commands.
    """

    @staticmethod
    def _process_field_value(field_value: Any) -> str:
        """
        Process a field value for string representation.

        :param field_value: The value to process.
        :return: A string representation of the value.
        """
        if isinstance(field_value, Path):
            return f'"{field_value.as_posix()}"'
        return str(field_value)

    def __str__(self) -> str:
        """
        Generate a string representation of the command, including its fields.

        :return: A formatted string representing the command.
        """
        fields = self.model_dump(exclude_none=True)
        cmd_name = fields.pop("name")
        return f"#{cmd_name}: {' '.join(map(Command._process_field_value, fields.values()))}"


class StackCommand(BaseCommand):
    """
    Base class for commands that can contain multiple subcommands.
    """

    def __str__(self) -> str:
        """
        Generate a string representation of the stack command, including its subcommands.

        :return: A formatted string representing the stack command.
        """
        fields = self.model_fields
        with StringIO() as str_buffer:
            for field_name, field in fields.items():
                field_value = getattr(self, field_name)
                if isinstance(field_value, (Command, StackCommand)):
                    str_buffer.write(str(field_value))
                    str_buffer.write("\n")
            out_str = str_buffer.getvalue().strip()
        return out_str


class CommandParser:
    """
    A class to parse command line arguments and return a Command instance.
    """

    commands_registry: Dict[str, Command] = {}

    @classmethod
    def register(cls, cmd_name: str) -> Callable:
        """
        Register a command class under a specific command name.

        :param cmd_name: The name under which the command will be registered.
        :return: A decorator that wraps the command class.
        """
        def wrapper(command_wrapped_class: Command) -> BaseModel:
            if cmd_name in cls.commands_registry:
                logger.debug(
                    f"A Command with name {cmd_name} is already registered; it will be overridden."
                )
            command_wrapped_class = patch_model(command_wrapped_class, name=(str, Field(default=cmd_name)))
            cls.commands_registry[cmd_name] = command_wrapped_class
            return command_wrapped_class
        return wrapper

    @classmethod
    def parse(cls, cmd_str: str) -> Command | StackCommand  | None:
        """
        Parse a command string and return an instance of the corresponding Command.

        :param cmd_str: The command string to parse.
        :return: An instance of the Command class corresponding to the parsed command.
        :raises NotImplementedError: If the command name is not registered.
        """
        match = re.search(r"#(\w+):\s(.+)", cmd_str)
        assert match is not None, f"Command string '{cmd_str}' is not valid"
        cmd_name = match.group(1).lower()
        cmd_args = match.group(2).split()

        if cmd_name not in cls.commands_registry:
            raise NotImplementedError(f"{cmd_name} not supported")

        cmd_class = cls.commands_registry[cmd_name]
        logger.debug(f"Using {cmd_class.__name__} to parse command '{cmd_name}'")

        # Special case for title command
        if cmd_name == "title":
            cmd_args = [match.group(2)]

        cmd_fields = dict(zip(cmd_class.model_fields, [cmd_name] + cmd_args))
        cmd = cmd_class(**cmd_fields)
        return cmd
