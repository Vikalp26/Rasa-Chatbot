from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

from bot_server_channel import BotServerInputChannel

from rasa_core.agent import Agent
from rasa_core import utils
from rasa_core.interpreter import RasaNLUInterpreter

interpreter = RasaNLUInterpreter('./models/nlu/default/weathernlu')

agent = Agent.load('models/dialogue', interpreter=interpreter)

channel = BotServerInputChannel(agent)

agent.handle_channel(channel)