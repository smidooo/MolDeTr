""" Normalization Parameters for the Target Data."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from moldetr.config import RegParamIndices


def permutation_invariant_coupling_constant_embedding(coupling_constant_in_points: list[float]):
    """Returns the embedding of the coupling constant."""
    return [np.sum(coupling_constant_in_points), np.min(coupling_constant_in_points),
     np.max(coupling_constant_in_points), np.std(coupling_constant_in_points)]
@dataclass(frozen=True)
class NormalizationParams:
    """Normalization Parameters for the Target Data. Contains the extrema of the regression parameters. The extrema are calculated by the calculate_extrema method. The extrema are saved in a json file. The extrema are loaded by the get_extrema method. If the extrema file does not exist, the calculate_extrema method is called.

    Attributes:
    _data_dir (Path): Path to the data directory.
    _extrema_file (str): Name of the extrema file.
    _number_of_samples (int): Number of samples.
    _reg_param_indices (dict): Dictionary containing the indices of the regression parameters.
    extrema (dict): Dictionary containing the extrema of the regression parameters.
    _extrema_path (Path): Path to the extrema file.
    """

    _data_dir: Path
    _extrema_file: str
    _input_length: int
    _reg_param_indices: RegParamIndices = field(default_factory=RegParamIndices)
    extrema: dict = field(init=False, default_factory=dict)
    _extrema_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Sets the path to the extrema file and loads the extrema."""
        object.__setattr__(self, "_extrema_path", self._data_dir / self._extrema_file)
        object.__setattr__(self, "extrema", self.get_extrema())

    def get_extrema(self) -> dict:
        """Returns the extrema of the regression parameters. If the extrema file does not exist, the calculate_extrema method is called."""
        if not Path(self._extrema_path).is_file():
            self.calculate_extrema()
        with open(self._extrema_path, "r") as fp:
            extrema = json.load(fp)
        return extrema

    def calculate_extrema(self) -> None:
        """Calculates the extrema of the regression parameters. The extrema are saved in a json file."""
        extrema_dict = {member: [float('inf'), float('-inf')] for member in self._reg_param_indices}
        print("Collecting Extrema...")

        # # Define the midpoint of the spectrum
        # midpoint_of_spectrum = (self._input_length - 1) / 2
        # max_deviation_from_midpoint = 0
        # Initialize a counter for the print statement
        file_counter = 0

        # Use pathlib to iterate over npz files in the directory
        for npz_file in self._data_dir.glob("*.npz"):
            file_counter += 1

            # Print progress after every 1000 files
            if file_counter % 10000 == 0:
                print(f"{file_counter} files processed.")

            with np.load(npz_file, allow_pickle=True) as file:
                for multiplet in file["labels"]:

                    # label = json.loads(str(multiplet).replace("'", '"'))
                    label=multiplet
                    center_position_in_points = label["center_position_in_points"]
                    line_width_in_points = label["line_width_in_points"]
                    bounding_box_range_in_points = label["bounding_box_range_in_points"]
                    coupling_constant_embeddings = permutation_invariant_coupling_constant_embedding(
                        label["coupling_constants_in_points"]
                    )





                    param_values = [
                        center_position_in_points,
                        line_width_in_points,
                        bounding_box_range_in_points,
                        coupling_constant_embeddings[0],
                        coupling_constant_embeddings[1],
                        coupling_constant_embeddings[2],
                        coupling_constant_embeddings[3],
                    ]
                    for key, param_value in zip(
                            self._reg_param_indices.keys(), param_values
                    ):
                        if key != "center_position_in_points":
                            if param_value < extrema_dict[key][0]:
                                extrema_dict[key][0] = param_value
                            if param_value > extrema_dict[key][1]:
                                extrema_dict[key][1] = param_value

        extrema_dict["center_position_in_points"] = [
            0,
            self._input_length-1
        ]

        # Save the extrema_dict to a JSON file
        with open(self._extrema_path, "w") as fp:
            json.dump(extrema_dict, fp)
        print("Saved Extrema.")
