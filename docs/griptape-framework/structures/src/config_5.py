from griptape.config import config
from griptape.config.drivers import AnthropicDriversConfig
from griptape.structures import Agent

config.drivers_config = AnthropicDriversConfig()

agent = Agent()
