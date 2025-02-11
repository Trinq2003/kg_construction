from base_classes.configuration import Configuration

class ERExtractorConfiguration(Configuration):
    def __init__(self):
        super().__init__()

    def _init_properties(self):
        return [
            ['prompt.yaml_path', 'llm/extraction_prompt_template.yaml', str]
        ]

# class SummarizerConfiguration(Configuration):
#     def __init__(self):
#         super().__init__()

#     def _init_properties(self):
#         return [
#             ['summary.summary_level', '', str],
#             ['summary.content_type', '', str],
#         ]