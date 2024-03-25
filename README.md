<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# 🔎 Prodigy-evaluate

This repository contains a Prodigy plugin for recipes to evaluate spaCy pipelines. It features multiple recipes:

1. `evaluate.evaluate`: Evaluate a spaCy pipeline on one or more datasets for different components. Passing flags like `--label-stats` or `--confusion-matrix` will compute a variety of evaluation metrics, including precision, recall, F1, accuracy, and more. 

<p align="center">
  <img src="images/evaluate.gif" width="75%">
</p>

2. `evaluate.evaluate-example`: Evaluate a spaCy pipeline on one or more datasets for different components on a **per-example basis**. This is helpful for debugging and for understanding the hardest examples for your model. 

<p align="center">
  <img src="images/evaluate_example.gif" width="75%">
</p>

3. `evaluate.nervaluate`: Evaluate a spaCy NER component on one or more datasets. This recipe uses the `nervaluate` library to calculate various metric for NER. You can learn more about the metrics in the [nervaluate documentation](https://github.com/MantisAI/nervaluate). This is helpful because the approach takes into account partial matches, which may be a more relevant metric for your NER use case. 

<p align="center">
  <img src="images/nervaluate.gif" width="75%">
</p>


You can install this plugin via `pip`. 

```
pip install "prodigy-evaluate @ git+https://github.com/explosion/prodigy-evaluate"
```

To learn more about this plugin and additional functionality, you can check the [Prodigy docs](https://prodi.gy/docs/plugins/#evaluate).

## Issues? 

Are you have trouble with this plugin? Let us know on our [support forum](https://support.prodi.gy/) and we'll get back to you! 