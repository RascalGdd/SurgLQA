"""Dataset registration for SurgLQA fine-tuning.

The dataset itself is intentionally not included in this repository. Configure
its locations through ``SURGLQA_ANNOTATION`` and ``SURGLQA_MEDIA_ROOT`` before
launching training.
"""

import os
import re


data_dict = {
    "surglqa_train": {
        "annotation_path": os.environ.get("SURGLQA_ANNOTATION", ""),
        "data_path": os.environ.get("SURGLQA_MEDIA_ROOT", ""),
    },
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name not in data_dict:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = data_dict[dataset_name].copy()
        missing = [key for key in ("annotation_path", "data_path") if not config[key]]
        if missing:
            raise ValueError(
                "SurgLQA data is not bundled with this release. Set "
                "SURGLQA_ANNOTATION and SURGLQA_MEDIA_ROOT before training."
            )
        config["sampling_rate"] = sampling_rate
        config_list.append(config)
    return config_list


if __name__ == "__main__":
    print(data_list(["surglqa_train"]))
