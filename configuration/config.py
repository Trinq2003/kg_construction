import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

with open('prompt_templates.yaml', 'r') as f:
    PROMPT_TEMPLATES = yaml.safe_load(f)