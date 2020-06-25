How to contribute?
==================

Some recommended packages for Python development
------------------------------------------------

* `pytest <https://docs.pytest.org/en/latest/>`_: Run :code:`pytest` in the main
  folder of the repository to run all :file:`test_*.py` files

* `pylint <https://pypi.org/project/pylint/>`_: Scan your code for naming
  conventions and proper use of Python

* `rope <https://github.com/python-rope/rope>`_: Python refactoring tools

* `sphinx <https://www.sphinx-doc.org/>`_: Generate documentation of your
  Python package

* `doc8 <https://pypi.org/project/doc8/>`_: A style checker for
  `reStructuredText
  <https://docutils.sourceforge.io/docs/ref/rst/introduction.html>`_

These packages and more can be installed using the `requirements-dev.txt
<https://github.com/ComPWA/expertsystem/blob/master/requirements-dev.txt>`_
file:

.. code-block:: shell

  pip install -r requirements_dev.txt


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

Python
------

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

  .. code-block: python

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
