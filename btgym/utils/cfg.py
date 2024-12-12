import os


##############
# Constants
##############

ROOT_PATH = os.path.abspath(os.path.join(__file__, "../.."))
ASSETS_PATH = os.path.join(ROOT_PATH, "assets")

OUTPUTS_PATH = os.path.join(ROOT_PATH, "../outputs")
os.makedirs(OUTPUTS_PATH, exist_ok=True)

TESTS_PATH = os.path.join(ROOT_PATH, "../tests")
os.makedirs(TESTS_PATH, exist_ok=True)

SUBGOAL_NET_PATH = os.path.join(ROOT_PATH, "../examples/training/subgoal_net.pth")





##############
# Variables
##############

llm_model = "gpt-4o"
# llm_model = "claude"
# llm_model = "claude-3-5-sonnet-20241022"
llm_temperature = 0.5
# llm_max_tokens = 2048

llm_embedding_model = "text-embedding-3-small"

