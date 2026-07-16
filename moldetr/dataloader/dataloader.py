"""
This module contains the dataloader for the data.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastai.vision.all import *
from torch.utils.data import Dataset  # DataLoader

from moldetr.config import RegParamIndices
from moldetr.dataloader.data_augmentation import DataAugmentationPartial
from moldetr.dataloader.normalization import permutation_invariant_coupling_constant_embedding
from moldetr.dataloader.transforms import Transform


def scale_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Scales the spectrum to the range [min, max]."""
    # Ensure the spectrum is not entirely zero
    assert np.any(spectrum != 0), "Spectrum contains only zero values. Check your data."

    return (spectrum)  / (np.max(spectrum)+1e-6)
@dataclass()
class DataReader(Dataset):
    """
    This class is used to read the data from the npz files. It is used by the dataloader. The data is normalized by the normalization parameters.

    Attributes:
        _data_dir (Path): Path to the data directory.
        _len (int): Number of samples.
        _max_num_multipelts (int): Maximum number of multiplets.
        _num_classes (int): Number of classes.
        transformation (Transform): Transformation object.
        reg_param_indices (dict): Dictionary containing the indices of the regression parameters.

    """

    _data_dir: Path
    _len: int
    _num_classes: int
    data_augmentation: Optional[DataAugmentationPartial]
    transformation: Transform
    reg_param_indices: RegParamIndices = field(default_factory=RegParamIndices)


    def __len__(self) -> int:
        """Returns the number of samples in the data set."""
        return self._len

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the sample at index idx. The sample is a tuple of the features and the targets. The features are normalized to maximum amplitude=1. The targets are normalized by the extrema of the regression parameters."""
        try:
            with np.load(self._data_dir / (str(idx) + ".npz"), allow_pickle=True) as sample:
                features = sample["spec"]
                # complex conjugate
                # features=np.conj(features)

                labels = sample['labels']

                # print(labels)
                # plot_spectrum(sample)
                # Wait for a condition or some event to signal shutdown

                # # Coin toss to decide if the spectrum should be flipped
                flip_spectrum = random.choice([True, False])
                # flip_spectrum = True
                # # Coin toss to decide if the spectrum should be cropped
                # # crop_spectrum = random.choice([True, False])
                crop_spectrum = True
                # features = scale_spectrum(features)
                if self.data_augmentation is not None:
                        # features = self.data_augmentation(features,labels) # for plotting distortions
                        features = self.data_augmentation(features)



                features = scale_spectrum(features)
                features = np.real(features[None, :])

                multiplet_list = []
                for label in labels:
                    multiplet = []
                    if label["proton_number"] !=6:
                        multiplet.append(label["proton_number"] - 1)
                    else:
                        multiplet.append(label["proton_number"]-2)



                    multiplet.append(

                            label["center_position_in_points"]/(len(features[0])-1)

                    )
                    multiplet.append(
                        self.transformation.transform(
                            label["line_width_in_points"],
                            self.reg_param_indices["line_width_in_points"],
                        )
                    )


                    multiplet.append(
                        self.transformation.transform(
                            label["bounding_box_range_in_points"],
                            self.reg_param_indices["bounding_box_range_in_points"],
                        )
                    )

                    # Apply various permutation invariant functions
                    for i,embedded_cc in enumerate(permutation_invariant_coupling_constant_embedding(label["coupling_constants_in_points"])):  # Assuming a maximum of 4 coupling constants
                        multiplet.append(
                            self.transformation.transform(
                                embedded_cc,
                                self.reg_param_indices[f"coupling_constant_{i + 1}_in_points"],
                            )
                        )




                    multiplet_list.append(multiplet)

            return (
                torch.tensor(features).float(),
                torch.tensor(multiplet_list).float(),
            )
        except zipfile.BadZipFile:
            print("Error at index: ", idx)

            error_file_path = "error_files.txt"
            existing_errors = set()

            # Read existing errors if the file already exists
            if os.path.exists(error_file_path):
                with open(error_file_path, "r") as error_file:
                    existing_errors = set(error_file.read().splitlines())

            # If the current error is new, write it to the file
            if str(idx) not in existing_errors:
                with open(error_file_path, "a") as error_file:
                    error_file.write(f"{idx}\n")

            return self.__getitem__(idx + 1)  # Proceed to the next item

def plot_spectrum(sample):
    features = sample["spec"]
    labels = sample["labels"]

    real_features=np.real(features)
    # Scaling the spectrum
    max_intensity = max(real_features)
    scaled_features = real_features / max_intensity

    # Determine the frequency axis
    frequency_hz = np.linspace(0, 1200, num=len(features))

    # Identify the range that contains signals
    min_signal_hz = 1200  # Start with the highest possible frequency
    max_signal_hz = 0     # Start with the lowest possible frequency

    for label in labels:
        center_position_hz = label["center_position_in_points"] / (len(features) - 1) * 1200
        range_hz = label["bounding_box_range_in_points"] / (len(features) - 1) * 1200  # Adjusted to Hz
        signal_start_hz = center_position_hz - range_hz / 2
        signal_end_hz = center_position_hz + range_hz / 2

        # Update the minimum and maximum signal bounds
        min_signal_hz = min(min_signal_hz, signal_start_hz)
        max_signal_hz = max(max_signal_hz, signal_end_hz)

    # Add a margin of 50 Hz to both sides
    margin_hz =(max_signal_hz - min_signal_hz) * 0.1  # 10% of the signal range
    min_signal_hz = max(min_signal_hz - margin_hz, 0)  # Ensure we don't go below 0 Hz
    max_signal_hz = min(max_signal_hz + margin_hz, 1200)  # Ensure we don't exceed 1200 Hz

    # Create the plot with a wider aspect ratio
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)  # Width is 12 inches and height is 4 inches

    # Plot the spectrum within the signal bounds
    ax.plot(frequency_hz, scaled_features, label='Spectrum', color='blue')

    # LaTeX caption variables
    caption_details = []
    signal_count = 1  # Counter for signal naming

    # Adding enhanced labels from the 'labels' dictionary
    for label in labels:
        center_position_hz = label["center_position_in_points"] / (len(features) - 1) * 1200
        line_width_hz = label["line_width_in_points"] / (len(features) - 1) * 1200  # Adjusted to Hz
        range_hz = label["bounding_box_range_in_points"] / (len(features) - 1) * 1200  # Adjusted to Hz
        coupling_constants = label["coupling_constants_in_points"] * (1/(len(features) - 1) * 1200) # Assuming already in Hz

        # Plotting signal details
        ax.axvline(x=center_position_hz, color='red', linestyle='--', label='Chemical Shift' if signal_count == 1 else "")
        ax.axvspan(center_position_hz - range_hz / 2, center_position_hz + range_hz / 2, color='gray', alpha=0.3, label='Signal Region' if signal_count == 1 else "")

        # Construct LaTeX caption part for this signal
        coupling_str = ', '.join([f'{cc:.2f} Hz' for cc in coupling_constants])
        caption_details.append(
            f"Signal {signal_count}: Protons {label['proton_number']}, Chemical Shift {center_position_hz:.2f} Hz, Line Width {line_width_hz:.2f} Hz, Signal Region {range_hz:.2f} Hz, Couplings [{coupling_str}].")
        signal_count += 1

    # Set labels and title
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title('Simulated Spectrum with Added Distortions')

    # Set x-axis limits to focus on the signal region only, with margin
    ax.set_xlim(min_signal_hz, max_signal_hz)
    ax.invert_xaxis()  # Invert x-axis to display increasing frequency from left to right



    ax.legend()  # Display the legend

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Display LaTeX caption
    caption = "\n".join(caption_details)
    print("Caption:")
    print(caption)

def process_spectrum_with_peak_based_cropping(features, labels, safety_margin=300, target_length=None, flip_spectrum=False):
    peak_positions = [int(label['center_position_in_points']) for label in labels]

    # Assuming peak_positions, safety_margin, and features are predefined
    # Calculate initial safe crop boundaries based on peak positions and safety margin
    safe_min_peak = max(min(peak_positions) - safety_margin, 0)  # Ensure it's not negative
    safe_max_peak = min(max(peak_positions) + safety_margin,
                        len(features) - 1)  # Ensure it doesn't exceed the spectrum length

    # Introduce variability within safe boundaries
    # For min_peak: Randomly choose between 0 and safe_min_peak, if there's room for variability
    min_peak = np.random.randint(0, safe_min_peak + 1) if safe_min_peak > 0 else 0

    # For max_peak: Randomly choose between safe_max_peak and len(features) - 1, if there's room for variability
    # Adjusting the range to ensure it doesn't exceed the last valid index
    max_peak = np.random.randint(safe_max_peak, len(features)) if safe_max_peak < len(features) - 1 else len(
        features) - 1

    # Additional check to ensure we don't unnecessarily cut the spectrum
    # If max_peak is the last possible index, we keep it to avoid unnecessary cutting
    max_peak = min(max_peak, len(features) - 1)

    # Crop the spectrum
    features= features[min_peak:max_peak]

    # Adjust label positions for cropping
    for label in labels:
        label['center_position_in_points'] -= min_peak

    # Calculate total padding needed to achieve target length
    if target_length is not None:
        padding_needed = target_length - len(features)
        if padding_needed > 0:
            padding_left = np.random.randint(0, padding_needed + 1)
            padding_right = padding_needed - padding_left
            features = np.pad(features, (padding_left, padding_right), 'constant')

            # Adjust label positions for left padding
            for label in labels:
                label['center_position_in_points'] += padding_left

    assert len(features) == target_length, f"Length of features is {len(features)}, but should be {target_length}."
    # Optionally flip the spectrum and adjust label positions accordingly
    if flip_spectrum:
        features = np.flip(features).copy()
        for label in labels:
            label['center_position_in_points'] = len(features) - label['center_position_in_points'] - 1

    features = np.real(features[None, :])

    return features, labels


class CustomDataLoader(DataLoader):
    def __init__(self, *args, batches_per_epoch=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        batch_count = 0
        for b in super().__iter__():
            yield b
            batch_count += 1
            if self.batches_per_epoch and batch_count >= self.batches_per_epoch:
                break




def get_num_samples(data_dir) -> int:
    """Returns the number of samples in the data directory."""
    initial_count = 0
    for path in data_dir.glob("*.npz"):
        if path.is_file():
            initial_count += 1
    return initial_count


def custom_collate(batch) -> tuple[torch.Tensor, dict]:
    """Returns a tuple of the features and the targets. The features are a tensor of shape (batch_size, 1, spectrum_length). The targets are a dictionary containing the targets and the number of targets."""
    features = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    len_targets = [item[1].size()[0] for item in batch]
    return (features, {"targets": targets, "num_targets": len_targets})


def get_train_test_set(
    data_dir: Path,
    batch_size: int,
    num_classes: int,
    reg_param_indices: RegParamIndices,
    num_workers: int,
    transformation: Transform,
    samples_per_epoch: Optional[int],
    data_augmentation: Optional[DataAugmentationPartial],
    specific_evaluation_set: Optional[Path]=None,
    test: bool = False,
) -> tuple[DataLoaders, DataLoaders]:
    """Returns the train, validation and test set."""


    if specific_evaluation_set and test:
        data_dir=Path(specific_evaluation_set)

    number_of_samples = get_num_samples(data_dir)
    dataset = DataReader(
        _data_dir=data_dir,
        _len=number_of_samples,
        _num_classes=num_classes,
        data_augmentation=data_augmentation,
        transformation=transformation,
        reg_param_indices=reg_param_indices,
    )

    if specific_evaluation_set and test:
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset,
            [
                int(0.0 * number_of_samples),
                int(0.0 * number_of_samples),
                int(1.0 * number_of_samples),
            ],
            generator=torch.Generator().manual_seed(42),
        )


    else:
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset,
            [
                int(0.92 * number_of_samples),
                int(0.06 * number_of_samples),
                int(0.02 * number_of_samples),
            ],
            generator=torch.Generator().manual_seed(42),
        )
    dls_train = DataLoader(
        train_set,
        reg_param_indices,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        create_batch=custom_collate,
        drop_last=True,

    )  # .cuda()


    dls_train=CustomDataLoader(dls_train,batches_per_epoch=samples_per_epoch//batch_size)
    dls_val = DataLoader(
        val_set,
        reg_param_indices,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        create_batch=custom_collate,
        drop_last=True,
    )
    dls_val=CustomDataLoader(dls_val, batches_per_epoch=samples_per_epoch//(batch_size))

    dls_train_val = DataLoaders(dls_train, dls_val)


    dls_test = DataLoaders(
        DataLoader(
            test_set,
            # reg_param_indices,
            batch_size=batch_size,
            num_workers=num_workers,
            create_batch=custom_collate,
            drop_last=True,
        )
    )

    return dls_train_val, dls_test
