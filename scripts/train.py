"""
This file contains the main function for training the model
"""

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from moldetr.config import MultipletConfig
from moldetr.learner.multi_multiplet_learner import init_learner
from moldetr.reproducibility import set_seed

# create config object for the model and training process and store it in the config store
cs = ConfigStore.instance()
cs.store(name="multiplet_config", node=MultipletConfig)




# Hydra Configuration Loading
@hydra.main(config_path='../conf', config_name='config_big')
def main(cfg: MultipletConfig) -> None:
    """
    main function for training the model

    Parameters
    ----------
    cfg: MultipletConfig object containing all parameters for training and testing the model as well as the data set to be used for training and testing the model


    Returns
    -------
    """

    print(OmegaConf.to_yaml(cfg))
    set_seed(42)
    learner = init_learner(cfg, test=False)

    learner.load(
        cfg.lognames.best_model_file,
        # "model_spin_system_ABCDEFG",
        # with_opt=True,
    )

    with learner.no_bar():
        learner.fit(cfg.optim_params.n_epochs_max)



if __name__ == "__main__":
    main()
