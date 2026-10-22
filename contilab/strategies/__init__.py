"""Built-in strategies register themselves on import."""

from contilab.strategies import naive  # noqa: F401
from contilab.strategies import ewc, lwf  # noqa: F401,E402
from contilab.strategies import derpp, er  # noqa: F401,E402
from contilab.strategies import agem  # noqa: F401,E402
from contilab.strategies import lora_per_task  # noqa: F401,E402
