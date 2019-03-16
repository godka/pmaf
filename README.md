Despite the abuandance of recently proposed models, these QoE models still lack the ablities on depicting the REAL performance of ABR algorithms. In this work, we aim to generate an ABR model without traditional certained QoE metrics. 

In details, The RL model trains a neural network for picking next chunk bitrate At from current network status St and the model aims to make {St+At} closer to the expert trajectory. Next, given a batch which consists of a group of expert trajectory which represented as the GOOD ABR sample that collected from the real world envoriments and {St+At} that the RL model generates, the discriminator aims to identifing which one is the real ABR trajectory. The output value from the disc network stands for the RL model's reward.

To train this, type

```
python train.py
```

for fun.

This work is still under constuction.
