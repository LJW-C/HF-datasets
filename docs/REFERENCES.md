# References

## Main Standard Reference

### ITU-R F.1487

ITU-R F.1487 is the recommended standard reference for testing HF modems with ionospheric channel simulators.

Link:

```text
https://www.itu.int/rec/R-REC-F.1487/en
```

Recommended wording:

```text
The HF fading channel follows the Watterson Gaussian scatter model and standard channel profiles described in ITU-R F.1487.
```

## Earlier / Historical Reference

### ITU-R F.520 / CCIR 520

ITU-R F.520 / CCIR 520 is an earlier recommendation on the use of high-frequency ionospheric channel simulators. It can be cited as historical background or for compatibility with older Watterson channel terminology.

Link:

```text
https://www.itu.int/rec/R-REC-F.520
```

## Related Implementation Reference

### PathSim

PathSim is an HF ionospheric Watterson channel simulator.

Link:

```text
https://github.com/bubnikv/pathsim
```

This repository does not directly execute PathSim. Instead, the Python fading channel implementation follows a PathSim-style approach:

```text
complex Gaussian noise
→ Gaussian FIR filtering
→ Watterson fading tap
```

## Propagation Prediction Reference

### ITU-R-HF

ITU-R-HF is a propagation prediction project based on ITU-R P.533 and P.372. It is useful for estimating realistic HF path parameters, but it is not a direct IQ/WAV fading channel simulator.

Link:

```text
https://github.com/ITU-R-Study-Group-3/ITU-R-HF
```

## Suggested Paper Description

English:

```text
The dataset was generated using a Python pipeline. The HF fading impairment was implemented as a PathSim-style Watterson Gaussian scatter channel, with channel profiles configured according to ITU-R F.1487/F.520.
```

Chinese:

```text
数据集由 Python 流程生成，其中 HF 衰落部分采用参考 PathSim 实现思想的 Watterson Gaussian scatter 信道，并按照 ITU-R F.1487/F.520 设置标准信道参数。
```

