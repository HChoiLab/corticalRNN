<figure>
  <img src="/images/iclr-fig1-schematic-colour-mod-crop.png" alt="Architecture of the CorticalRNN">
  <figcaption style="text-align: center;">Setup and architecture of the CorticalRNN.</figcaption>
</figure>

## Exploring the Architectural Biases of the Canonical Cortical Microcircuit

All architecture files are provided as .py files in the "architectures" folder. These include:
- CorticalRNN (architecture_original)
- No Feedback (architecture_no_feedback)
- Bi-directional Feedback (architecture_bidirectional_feedback)
- Uni-directional Feedback (architecture_single_feedback)
- Population controlled (architecture_same_pop)
- No Time Delays (architecture_no_td_all_combo)

All task implementations can be found in the file generate_data.py

The folder "recon-loss" trains the architectures using the reconstruction loss while the folder "pc-loss" trains the architectures using two-phase training which uses the predicitive-coding loss.
- Notebooks to run the architectures on the different tasks are provided in the folders pertaining to the task within each folder.

When running the notebooks, make sure the notebook itself, the data_generator, and architecture file are all in the same folder.

The folders also contain a notebook "plot-generator" that produces all plots corresponding to the task.
- Do not run the plot generator unless the other notebooks have been run first.
- Make sure to add the appropriate path in a notebook when running.

The notebook "no-time-delay-microcircuit" trains the microcircuit without time delays on the sequence memorization task.
