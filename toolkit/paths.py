import os

TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ROOT = os.path.join(TOOLKIT_ROOT, 'config')
KEYMAPS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "keymaps")
ORIG_CONFIGS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "orig_configs")
DIFFUSERS_CONFIGS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "diffusers_configs")
COMFY_MODELS_PATH = None

# The configured models folder lives in the toolkit's .env (MODELS_PATH=...).
# Read it HERE instead of relying on the entrypoint to have called
# load_dotenv() first: anything that imports this module -- run.py, a script,
# a tool, a test, a hand-run snippet -- must see the same folder. When this
# only read os.environ, every path that did not arrive through run.py (which
# does load .env) silently fell back to the repo-local default below and
# downloaded multi-GB components onto the wrong disk, next to copies that
# already existed in the real folder.
#
# load_dotenv does not overwrite a variable that is already set, so an
# explicitly exported MODELS_PATH still wins over the file, and a machine
# with no .env keeps the repo-local default.
from dotenv import load_dotenv  # noqa: E402  (needs TOOLKIT_ROOT above)

load_dotenv(os.path.join(TOOLKIT_ROOT, ".env"))

# check if ENV variable is set
if 'MODELS_PATH' in os.environ and os.environ['MODELS_PATH'].strip() != "":
    MODELS_PATH = os.environ['MODELS_PATH']
else:
    MODELS_PATH = os.path.join(TOOLKIT_ROOT, "models")


def get_path(path):
    # we allow absolute paths, but if it is not absolute, we assume it is relative to the toolkit root
    if not os.path.isabs(path):
        path = os.path.join(TOOLKIT_ROOT, path)
    return path
