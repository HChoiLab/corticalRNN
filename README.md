![CorticalRNN architecture](/images/iclr-fig1-schematic-colour-mod-crop.png)
<div style="text-align: center;">
  <em>Setup and architecture of the CorticalRNN</em>
</div>

### Abstract

<div style="text-align: justify;">
  The cortex plays a crucial role in various perceptual and cognitive functions, driven by its basic unit, the <em>canonical cortical microcircuit</em>.
  Yet, we remain short of a framework that definitively explains the structure-function relationships of this fundamental neuroanatomical motif.
  To better understand how physical substrates of cortical circuitry facilitate their neuronal dynamics, we employ a computational approach using recurrent neural networks and representational analyses.
  We examine the differences manifested by the inclusion and exclusion of biologically-motivated inter-areal laminar connections on the computational roles of different neuronal populations in the microcircuit of hierarchically-related areas, throughout learning.
  Our findings show that the presence of feedback connections correlates with the functional modularization of cortical populations in different layers, and provides the microcircuit with a natural inductive bias to differentiate expected and unexpected inputs at initialization, which we justify mathematically.
  Furthermore, when testing the effects of training the microcircuit and its variants with a predictive-coding inspired strategy, we find that doing so helps better encode noisy stimuli in areas of the cortex that receive feedback,
  all of which combine to suggest evidence for a predictive-coding mechanism serving as an intrinsic operative logic in the cortex.
</div>

### Instructions
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
