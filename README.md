Despite the abundance of recently proposed schemes, these QoE models still lack the abilities in depicting the REAL performance of ABR algorithms. In this work, we aim to generate an ABR model without traditional certain QoE metrics.

In details, The RL model trains a neural network for picking next chunk bitrate At from current network status St, and the model aims to make {St+At} closer to the expert trajectory. Next, given a batch which consists of a group of the expert trajectory which represented as the GOOD ABR sample that collected from the real world environments and {St+At} that the RL model generates, the discriminator aims to identify which one is the real ABR trajectory. The output value from the discriminator network stands for the RL model's reward.

To train this, type

```
python train.py
```

for fun.

This work is still under constuction.
