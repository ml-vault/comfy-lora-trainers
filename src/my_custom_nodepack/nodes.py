from .node_items import DataPackLoaderNode, ImageExporterNode, TrainLoraXlNode, TrainConfigNode, SamplerConfigNode, OutputConfigNode, DatasetLoaderNode, DatasetNode

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DataPackLoaderNode": DataPackLoaderNode,
    "ImageExporterNode": ImageExporterNode,
    "TrainLoraXlNode": TrainLoraXlNode,
    "TrainConfigNode": TrainConfigNode,
    "SamplerConfigNode": SamplerConfigNode,
    "OutputConfigNode": OutputConfigNode,
    "DatasetLoaderNode": DatasetLoaderNode,
    "DatasetNode": DatasetNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DataPackLoaderNode": "DataPack Loader Node",
    "ImageExporterNode": "Image Exporter Node",
    "TrainLoraXlNode": "Train LoRA XL Node",
    "TrainConfigNode": "Train Config Node",
    "SamplerConfigNode": "Sampler Config Node",
    "OutputConfigNode": "Output Config Node",
    "DatasetLoaderNode": "Dataset Loader Node",
    "DatasetNode": "Dataset Node"
}
