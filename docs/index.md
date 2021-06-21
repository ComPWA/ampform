# Welcome to AmpForm!

```{title} Welcome

```

[![PyPI package](https://badge.fury.io/py/ampform.svg)](https://pypi.org/project/ampform)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/ampform)](https://pypi.org/project/ampform)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ComPWA/ampform/blob/stable)
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/ampform/stable?filepath=docs/usage)

:::{margin}

The original project was the [PWA Expert System](https://expertsystem.rtfd.io).
AmpForm originates from its
[`amplitude`](https://expertsystem.readthedocs.io/en/stable/api/expertsystem.amplitude.html)
module.

:::

AmpForm aims to automatically formulate {doc}`PWA <pwa:index>` amplitude models
for arbitrary particle transitions. All allowed transitions between some
initial and final state are generated with the {mod}`qrules` package. You can
then use {mod}`ampform` to formulate these state transitions into an amplitude
model with some formalism, like the helicity formalism.

```{rubric} Some highlights

```

- **PWA made easy**<br> Need to add some resonance, remove one, or check out
  some coupled channel? {doc}`QRules <qrules:index>` has you covered! It finds
  all allowed transitions and AmpForm will formulate the amplitude model.

- **Narrow the gap between code and theory**<br> AmpForm amplitude models are
  formulated with {doc}`SymPy <sympy:index>`. For instance, have a look at
  {doc}`/usage/dynamics` to see how easy it is to inspect and understand the
  math as well as the code behind the model.

- **Adjust and modify the model to your needs**<br> Need some additional
  background terms some specific dynamics functions for your decay channel? Use
  {doc}`/usage/dynamics/custom`, substitute or add specific expressions, or
  couple parameters in an instant, without having to change the framework.

- **Convert to several computational back-ends**<br> The amplitude models can
  easily be converted to computational back-ends like
  [JAX](https://jax.readthedocs.io), [TensorFlow](https://www.tensorflow.org),
  and [NumPy](https://numpy.org). The {doc}`TensorWaves <tensorwaves:index>`
  package facilitates this and also provides tools to generate toy Monte Carlo
  data samples.

- **Use your favorite Python packages**<br> All this functionality is offered
  in an open and transparent style (see {doc}`API </api/ampform>`), so that you
  can navigate AmpForm's output and feed it to whatever other Python packages
  come in handy for you. For instance, check out {doc}`/usage/interactive` to
  see how to inspect your model with an interactive widget.

The {doc}`/usage` pages illustrate how to do some of these things. You can run
each of them as Jupyter notebooks with the {fa}`rocket` launch button in the
top-right corner ;) Enjoy!

```{rubric} Table of contents

```

```{toctree}
---
maxdepth: 3
---
install
usage
references
API <api/ampform>
Changelog <https://github.com/ComPWA/ampform/releases>
Upcoming features <https://github.com/ComPWA/ampform/milestones?direction=asc&sort=title&state=open>
Develop <https://pwa.readthedocs.io/develop.html>
```

- {ref}`Python API <modindex>`
- {ref}`General Index <genindex>`
- {ref}`Search <search>`

```{toctree}
---
caption: Related projects
hidden:
---
QRules <https://qrules.readthedocs.io>
TensorWaves <https://tensorwaves.readthedocs.io>
PWA Pages <https://pwa.readthedocs.io>
```
