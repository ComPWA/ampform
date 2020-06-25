How to contribute?
==================


Python developer tools
----------------------

The PWA Expert System repository comes with a set of Python developer tools.
They are defined in the `requirements-dev.txt
<https://github.com/ComPWA/expertsystem/blob/master/requirements-dev.txt>`_
file, which means you can install them all in one go with:

.. code-block:: shell

  pip install -r requirements_dev.txt

Most of the tools defined come with specific configuration files (e.g.
`pyproject.toml
<https://github.com/ComPWA/expertsystem/blob/master/pyproject.toml>`_ for
`black <https://black.readthedocs.io/>`_, `.pylintrc
<https://github.com/ComPWA/expertsystem/blob/master/.pylintrc>`_ for `pylint
<http://pylint.pycqa.org/en/latest/>`_, and `tox.ini
<https://github.com/ComPWA/expertsystem/blob/master/tox.ini>`__ for `flake8
<https://flake8.pycqa.org/>`_ and `pydocstyle <http://www.pydocstyle.org/>`_).
These config files **define our convention policies**. If you run into
persistent linting errors this may mean we need to further specify our
conventions. In that case, it's best to create an issue and propose a policy
change that can then be formulated in the config files.

All **style checks** are enforced through a tool called `pre-commit
<https://pre-commit.com/>`_. Upon committing, :code:`pre-commit` runs a set of
checks defined in the file `.pre-commit-config.yaml
<https://github.com/ComPWA/expertsystem/blob/master/.pre-commit-config.yaml>`_
over all staged files. You can also quickly run all checks over *all* indexed
files in the repository with the command:

.. code-block:: shell

  pre-commit run -a

This command is also run on Travis CI whenever you submit a pull request,
ensuring that all files in the repository follow the conventions set in the
config files of these tools.

More thorough checks (that is, **runtime tests**) can be run in one go with the
command

.. code-block:: shell

  tox

This command will run :code`pytest`, check for :ref:`test coverage
<contribute:Test coverage>`, verify the hyperlinks in documentation (requires
internet connection), and run the Jupyter notebooks. All this can take a few
minutes, but it's definitely worth it to *run tox before submitting a pull
request!*

The different :code:`tox` tests are defined in the `tox.ini
<https://github.com/ComPWA/expertsystem/blob/master/tox.ini>`__ file under
:code:`envlist`.


Test coverage
-------------

Try to keep test coverage high. You can compute current coverage by running

.. code-block:: shell

  pytest \
    --cov-report=html \
    --cov-report=xml \
    --cov=expertsystem

and opening :file:`htmlcov/index.html` in a browser. In VScode, you can
visualize which lines in the code base are covered by tests with the `Coverage
Gutters
<https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters>`_
extension (for this you need to run :code:`pytest` with the flag
:code:`--cov-report=xml`).


Documentation
-------------

The documentation that you find on `expertsystem.rtfd.io
<http://expertsystem.rtfd.io>`_ are built from the `documentation source code
folder <https://github.com/ComPWA/expertsystem/tree/master/doc>`_ (:file:`doc`)
with `Sphinx <https://www.sphinx-doc.org>`_. Sphinx also builds the API and
therefore checks whether the `docstrings
<https://www.python.org/dev/peps/pep-0257/>`_ in the Python source code are
valid and correctly interlinked.

If you followed the section :ref:`contribute:Python developer tools`, you can
quickly build the documentation from the root directory of this repository with
the command:

.. code-block:: shell

  tox -e doc

Alternatively, you can run :code:`sphinx-build` yourself. The requirements for
that are in the `doc/requirements.txt
<https://github.com/ComPWA/expertsystem/blob/master/doc/requirements.txt>`_
file:

.. code-block:: shell

  cd doc
  pip install -r requirements.txt
  make html

If you want to render the output of the `Jupyter notebook examples
<https://github.com/ComPWA/expertsystem/tree/master/examples>`_, set the
:code:`NBSPHINX_EXECUTE` environment variable first:

.. code-block:: shell

  NBSPHINX_EXECUTE= make html

Note that the notebooks are also run if you run :code:`tox`.

A nice feature of `Read the Docs <https://readthedocs.org/>`_, where we host
our documentation, is that documentation is built for each pull request as
well. This means that you can view the documentation for your changes as well.
For more info, see `here
<https://docs.readthedocs.io/en/stable/guides/autobuild-docs-for-pull-requests.html>`__,
or just click "details" under the RTD check once you submit your PR.


Git
---

* Please use
  `conventional commit messages <https://www.conventionalcommits.org/>`_: start
  the commit with a semantic keyword (see e.g. `Angular
  <https://github.com/angular/angular/blob/master/CONTRIBUTING.md#type>`_ or
  `these examples <https://seesparkbox.com/foundry/semantic_commit_messages>`_,
  followed by `a column <https://git-scm.com/docs/git-interpret-trailers>`_,
  then the message. The message itself should be in imperative mood — just
  imagine the commit to give a command to the code framework. So for instance:
  :code:`feat: add coverage report tools` or :code:`fix: remove ...`.

* In the master branch, each commit should compile and be tested. In your own
  branches, it is recommended to commit frequently (WIP keyword), but squash
  those commits upon submitting a merge request.


Python conventions
------------------

* Follow :pep:`8` conventions.

* Any Python file that's part of a module should contain (in this order):

  1. A docstring describing what the file contains and does, followed by two
  empty lines.

  2. A definition of `__all__
     <https://docs.python.org/3/tutorial/modules.html#importing-from-a-package>`_,
     so that you can see immediately what this Python file defines, **followed
     by two empty lines**.

  3. Only after these come the :code:`import` statements, following the
     :pep:`8` conventions for imports.

* When calling or defining multiple arguments of a function and multiple
  entries in a data container, split the entries over multiple lines and end
  the last entry with a comma, like so:

  .. code-block:: python

    __all__ = [
        'core',
        'optimizer',
        'physics',
        'plot',
    ]

  This is to facilitate eventual `diff <https://git-scm.com/docs/git-diff>`_
  comparisons in Git.


Visual Studio code
------------------

We recommend using `Visual Studio Code <https://code.visualstudio.com/>`_ as
it's free, regularly updated, and very flexible through it's wide offer of user
extensions.

If you add or open this repository to/as a `VSCode workspace
<https://code.visualstudio.com/docs/editor/multi-root-workspaces>`_, the
:file:`.vscode/settings.json` will ensure that you have the right developer
settings for this repository.

You can still specify your own settings in `either the user or encompassing
workspace settings <https://code.visualstudio.com/docs/getstarted/settings>`_,
as the VSCode settings that come with this are folder settings.
