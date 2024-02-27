<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# 🔎 Prodigy-evaluate

This repository contains a Prodigy plugin for recipes to evaluate spaCy pipelines. It features two recipes:

1. `evaluate`: Evaluate a spaCy pipeline on one or more datasets for different components. Per-component evaluation sets can be provided using the `eval:` prefix for consistency. Passing flags like `--label-stats` or `--confusion-matrix` will compute a variety of evaluation metrics, including precision, recall, F1, accuracy, and more. 

2. `evaluate-example`: Evaluate a spaCy pipeline on one or more datasets for different components on a **per-example basis**. This is helpful for debugging and for understanding the hardest examples for your model. 

Here's a preview of the `evaluate` recipe with some availabile parameters in the terminal:

<p align="center">
  <img src="images/evaluate.gif">
</p>

You can install this plugin via `pip`. 

```
pip install "prodigy-evaluate @ git+https://github.com/explosion/prodigy-evaluate"
```

To learn more about this plugin and additional functionality, you can check the [Prodigy docs](https://prodi.gy/docs/plugins/#evaluate).

## Issues? 

Are you have trouble with this plugin? Let us know on our [support forum](https://support.prodi.gy/) and we'll get back to you! 