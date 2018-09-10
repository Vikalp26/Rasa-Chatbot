from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.fallback import FallbackPolicy

if __name__ == '__main__':
	logging.basicConfig(level='INFO')

	training_data_file = './data/stories.md'
	model_path = './models/dialogue'

	fallback = FallbackPolicy(fallback_action_name="action_default_fallback",
                          core_threshold=0.9,
                          nlu_threshold=0.9)	
	
	agent = Agent('weather_domain.yml', policies = [MemoizationPolicy(max_history = 2), KerasPolicy(), fallback])
	
	agent.train(
			training_data_file,
			epochs = 400,
			batch_size = 15,
			validation_split = 0.2)
			
	agent.persist(model_path)