```{eval-rst}
.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:

```

# API

```{eval-rst}
.. module:: eykthyr

.. automodule:: eykthyr
   :noindex:
```

## IO: `io`

Tools for loading and saving Eykthyr data and parameters

```{eval-rst}
.. module:: eykthyr.model
.. currentmodule:: eykthyr

.. autosummary::
    :toctree: api/
    :recursive:

    model.save_anndata
    model.load_anndata
```

## Model

Entry points for implementations of the Eykthyr algorithm.

```{eval-rst}
.. module:: eykthyr.model
.. currentmodule:: eykthyr

.. autosummary::
    :toctree: api/
    :recursive:

    model.Eykthyr
```

## Preprocessing

Objects that are helpful for working with Eykthyr.

```{eval-rst}
.. module:: eykthyr.embedding
.. currentmodule:: eykthyr

.. autosummary::
    :toctree: api/
    :recursive:

    embedding.Embedding
```

## Analysis: `pl`

Functions for visualizing and evaluating Eykthyr results.

```{eval-rst}
.. module:: eykthyr.plotting
.. currentmodule:: eykthyr

.. autosummary::
    :toctree: api/
    :recursive:

    plotting.paga_spatial_simulation
    plotting.prep_paga
```
