from rasa_core.channels import HttpInputChannel
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_slack_connector import SlackInput


nlu_interpreter = RasaNLUInterpreter('./models/nlu/default/weathernlu')
agent = Agent.load('./models/dialogue', interpreter = nlu_interpreter)

input_channel = SlackInput('xoxp-411998793184-412854261061-412718450354-7b6b4b4676c93f1533c8faeb53f11085', #app verification token
							'xoxb-411998793184-413752901623-qtcDkC9Al5BCmBilUQ229SrB', # bot verification token
							'L3PIeWvoHbbJA5M1W5uJ59bO', # slack verification token
							True)

agent.handle_channel(HttpInputChannel(5004, '/', input_channel))