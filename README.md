## Meta-Learning on Analog Crossbar Arrays for Adaptation on the Edge

meta-analog is a minimalistic JAX implementation of the Model-Agnostic Meta-Learning ([MAML](https://arxiv.org/abs/1703.03400)) for Spiking Neural Networks 
realized on a differentiable analog hardware model. Sounds too fancy? Here is the reasoning behind it:

- **Why analog hardware?** Analog hardware e.g. PCM crossbar arrays can implement $$O(1)$$ matmul operation using Kirchhoff's current law and Ohm's Law. This leads to extremely low inference latency and energy cost, required by the edge computing. These devices are currently the most scalable solutions for the edge inference.

- **Why spikes?** Purely as an engineering requirement. On noisy substrates like analog crossbar arrays (or brain), the spikes are the most robust way to encode information. Spikes are also the language of event-based sensors.

- **Why meta-learning?** Meta-learning is silver bullet to edge adaptation as it solves multiple major issues of analog hardware and spiking networks, simultaneously.

    1) The memristive devices of the analog hardware has limited bit-resolution of ~2-3 bits. This leads to huge performance hits during the training. However, meta-learned inner loop updates have 100-1000x larger magnitudes than usual training scenarios.

    2) Efficient credit assignment in multilayer spiking networks is an unsolved problem. Meta-learning allows updating only the last layer of the network, which can be implemented simple Delta Rule.

    3) Training from tabula-rasa is not feasible for edge devices. Because offline training with larger datasets offer better optimization of the network. And can be programmed with simple iterative programming.

### meta-analog

There are three main components of the meta-analog:

- `train.py` : Implements meta-training and meta-testing of MAML for the spiking network.
- `network.py`: Implements 2-layer Leaky Integrate-and-Fire (LIF) spiking neural network.
- `analog.py` : Implements differentiable PCM crossbar array model.


### PCM Crossbar Model

<center>
<img src="img/model.png" width="100%"/>
<i>Implemented PCM model</i><br/>
</center>
<p></p>

PCM crossbar array simulation framework is developed based on the PCM device model introduced by [_Nandakumar, S. R. et al, 2018_](https://aip.scitation.org/doi/10.1063/1.5042408). This empirical model is based on the experimental data gathered from 10,000 devices and captures the programming noise, read noise and temporal conductance drift without diluting the analog device non-idealities (cycle-to-cycle and device-to-device variability). It is straightforward to deploy models trained in this PCM model to real hardware, see [_our paper with IBM Research_](https://ieeexplore.ieee.org/abstract/document/9869963).

### Example Usage

- To start MAML training using PCM-based analog weight implementation:

`python train.py --n_iter 50000`

- To start MAML training in the _performance_ mode using quantized 2-bit weight updates. This mode randomly initializes the weights but every inner loop weight update is quantized to 2-bit.

:  `python train.py --n_iter 50000 --perf`
