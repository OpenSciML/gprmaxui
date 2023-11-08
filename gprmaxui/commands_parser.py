from __future__ import annotations

import typing
from io import StringIO
from pathlib import Path

from pydantic import BaseModel
from typing import Callable
import logging
import re

logger = logging.getLogger("rich")


class BaseCommand(BaseModel):
    """
    Abstract class for command
    """

    def dict(self, **kwargs):
        """
        Override dict method to exclude hidden fields
        :param kwargs:
        :return:
        """
        hidden_fields = set(
            attribute_name
            for attribute_name, model_field in self.__fields__.items()
            if model_field.field_info.extra.get("hidden") is True
        )
        kwargs.setdefault("exclude", hidden_fields)
        return super().dict(**kwargs)

    def __call__(self, *args, **kwargs):
        """
        print the command
        :param args:
        :param kwargs:
        :return:
        """
        self.print()

    def print(self):
        """
        print the command
        :return:
        """
        print(self)


class Command(BaseCommand):
    """
    Abstract class for command
    """

    name: typing.Optional[str] = None

    @staticmethod
    def _process_fied_value(field_value):
        if isinstance(field_value, Path):
            return '"' + field_value.as_posix() + '"'
        return str(field_value)

    def __str__(self):
        fields = self.dict()
        cmd_name = fields.pop("name")
        return f"#{cmd_name}: {' '.join(map(Command._process_fied_value, fields.values()))}"


class StackCommand(BaseCommand):
    """
    Abstract class for multi command
    """

    def __str__(self):
        fields = self.__fields__
        with StringIO() as str_buffer:
            for field_name, field in fields.items():
                field_value = getattr(self, field_name)
                if isinstance(field_value, (Command, StackCommand)):
                    str_buffer.write(str(field_value))
                    str_buffer.write("\n")
            out_str = str_buffer.getvalue()
        out_str = out_str.strip()
        return out_str


class CommandParser:
    """
    A class to parse command line arguments and return a Command instance.
    """

    commands_registry = {}

    @classmethod
    def register(cls, cmd_name) -> Callable:
        def wrapper(command_wrapped_class: Command) -> Callable:
            if cmd_name in cls.commands_registry:
                logger.warning(
                    f"A Command with name {cmd_name} is already registered. it will be override"
                )
            cls.commands_registry[cmd_name] = command_wrapped_class
            # we need to set the default value of the name field to the command name
            command_wrapped_class.__fields__["name"].default = cmd_name
            return command_wrapped_class

        return wrapper

    @classmethod
    def parse(cls, cmd_str: str) -> Command:
        """Build a DataReader instance for the given extension.
        ext: str  - The extension of the file to read.
        kwargs: dict - The keyword arguments to pass to the DataReader constructor.
        :rtype: DataReader - The DataReader instance.
        """
        match = re.search(r"#(\w+):\s(.+)", cmd_str)
        assert match is not None, f"Command string {cmd_str} is not valid"
        cmd_name = match.group(1).lower()
        cmd_args = match.group(2).split()
        if cmd_name not in cls.commands_registry:
            raise NotImplementedError(f"{cmd_name} not supported")
        cmd_class = cls.commands_registry[cmd_name]
        logger.debug(f"using {cmd_class.__name__} to parse command {cmd_name}")
        # special case for title command
        if cmd_name == "title":
            cmd_args = [match.group(2)]
        cmd_fields = dict(zip(cmd_class.__fields__, [cmd_name] + cmd_args))
        cmd = cmd_class(**cmd_fields)
        return cmd
