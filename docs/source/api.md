```{eval-rst}
.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:

```

# API Reference

```{eval-rst}
.. module:: eykthyr

.. automodule:: eykthyr
   :noindex:
```

## Main class

The `Eykthyr` class is the primary entry point for the full analysis pipeline.

```{eval-rst}
.. module:: eykthyr.eykthyr
.. currentmodule:: eykthyr

.. autosummary::
    :toctree: api/

    eykthyr.Eykthyr
    eykthyr.load_anndata
```

## Visualization: `pl`

Functions for visualizing perturbation simulations and developmental scores.
Access these via `import eykthyr; eykthyr.pl.<function>`.

```{eval-rst}
.. module:: eykthyr.plotting
.. currentmodule:: eykthyr

.. autosummary::
    :toctree: api/

    plotting.prep_paga
    plotting.paga_spatial_simulation
    plotting.umap_spatial_simulations
    plotting.umap_spatial_simulation
    plotting.development_simulation
```

## Embedding container

```{eval-rst}
.. module:: eykthyr.embedding
.. currentmodule:: eykthyr

.. autosummary::
    :toctree: api/

    embedding.Embedding
```
