"""LeRobot-compatible batch keys (subset) — no lerobot dependency."""

OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"
OBS_LANGUAGE = OBS_STR + ".language"
OBS_LANGUAGE_TOKENS = OBS_LANGUAGE + ".tokens"
OBS_LANGUAGE_ATTENTION_MASK = OBS_LANGUAGE + ".attention_mask"

ACTION = "action"

# Default camera key for single-view teaching env
CAMERA_MAIN = f"{OBS_IMAGES}.main"
