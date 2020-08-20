.. cSpell:ignore aquirdturtle, docnb, htmlcov, ijmbarr, labextension, pylintrc, ryantam, serverextension, testenv

How to contribute?
==================

.. tip::

  Bugs can be reported `here
  <https://github.com/ComPWA/expertsystem/issues/new/choose>`__. Also, please
  do have a look at the `'good first issues' page
  <https://github.com/ComPWA/expertsystem/issues?q=is%3Aissue+is%3Aopen+label%3A%22%F0%9F%92%AB+Good+first+issue%22>`_:
  they are nice challenges to get into the find your way around the source
  code! ;)

If you have installed the `expertsystem` in :ref:`install:Development mode`, it
is easy to tweak the source code and try out new ideas immediately, because the
source code is considered the 'installation'.

.. note::

  The easiest way to contribute, is by using :ref:`Conda <install:Conda
  environment>` and :ref:`contribute:Visual Studio code`. In that case, the
  complete developer install procedure becomes:

  .. code-block:: shell

    git clone https://github.com/ComPWA/expertsystem.git
    cd expertsystem
    conda env create
    conda activate es
    pip install -e .[dev]
    code .  # open folder in VSCode

  For more info, see :ref:`contribute:Visual Studio code`.


Automated style checks
----------------------

When working on the source code of the `expertsystem`, it is highly recommended
to install certain additional Python tools. Assuming you installed the
`expertsystem` in :ref:`development mode <install:Development mode>`, these
additional tools can be installed into your :ref:`virtual environment
<install:Step 2: Create a virtual environment>` in one go:

.. code-block:: shell

  pip install -e .[dev]

Most of the tools that are installed with this command use specific
configuration files (e.g. `pyproject.toml
<https://github.com/ComPWA/expertsystem/blob/master/pyproject.toml>`_ for
`black <https://black.readthedocs.io/>`_, `.pylintrc
<https://github.com/ComPWA/expertsystem/blob/master/.pylintrc>`_ for `pylint
<http://pylint.pycqa.org/en/latest/>`_, and `tox.ini
<https://github.com/ComPWA/expertsystem/blob/master/tox.ini>`__ for `flake8
<https://flake8.pycqa.org/>`_ and `pydocstyle <http://www.pydocstyle.org/>`_).
These config files **define our convention policies**, such as :pep:`8`. If you
run into persistent linting errors, this may mean we need to further specify
our conventions. In that case, it's best to create an issue and propose a
policy change that can then be formulated in the config files.


Pre-commit
----------

All **style checks** are enforced through a tool called `pre-commit
<https://pre-commit.com/>`__. This tool needs to be activated, but only once,
after you clone the repository:

.. code-block:: shell

  pre-commit install

Upon committing, :code:`pre-commit` now runs a set of checks as defined in the
file `.pre-commit-config.yaml
<https://github.com/ComPWA/expertsystem/blob/master/.pre-commit-config.yaml>`_
over all staged files. You can also quickly run all checks over *all* indexed
files in the repository with the command:

.. code-block:: shell

  pre-commit run -a

This command is also run on GitHub actions whenever you submit a pull request,
ensuring that all files in the repository follow the conventions set in the
config files of these tools.


Testing
-------

More thorough checks (that is, **runtime tests**) can be run in one go with the
command

.. code-block:: shell

  tox

This command will run :code:`pytest`, check for test coverage, build the
documentation, and verify cross-references in the documentation and the API.
It's especially recommended to *run tox before submitting a pull request!*

More specialized :code:`tox` tests are defined in the `tox.ini
<https://github.com/ComPWA/expertsystem/blob/master/tox.ini>`__ file, under
each :code:`testenv`.

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

You can quickly build the documentation from the root directory of this
repository with the command:

.. code-block:: shell

  tox -e doc

Alternatively, you can run :code:`sphinx-build` yourself as follows:

.. code-block:: shell

  cd doc
  make html  # or NBSPHINX_EXECUTE= make html

A nice feature of `Read the Docs <https://readthedocs.org/>`_, where we host
our documentation, is that documentation is built for each pull request as
well. This means that you can view the documentation for your changes as well.
For more info, see `here
<https://docs.readthedocs.io/en/stable/guides/autobuild-docs-for-pull-requests.html>`__,
or just click "details" under the RTD check once you submit your PR.


Jupyter Notebooks
-----------------

The `examples <https://github.com/ComPWA/expertsystem/tree/master/examples>`_
folder contains a few notebooks that illustrate how to use the `expertsystem`.
These notebooks are also available on the :doc:`Usage <usage>` page and are run
and tested whenever you make a :ref:`pull request <contribute:Git and GitHub>`.
As such, they serve both as up-to-date documentation and as tests of the
interface.

If you want to improve those notebooks, we recommend working with `Jupyter Lab
<https://jupyterlab.readthedocs.io/en/stable/>`_, which is installed with the
:code:`dev` requirements of the `expertsystem`. Jupyter Lab offers a nicer
developer experience than the default Jupyter notebook editor does. In
addition, recommend to install a few extensions:

.. code-block:: shell

  jupyter labextension install jupyterlab-execute-time
  jupyter labextension install @ijmbarr/jupyterlab_spellchecker
  jupyter labextension install @aquirdturtle/collapsible_headings
  jupyter labextension install @ryantam626/jupyterlab_code_formatter

  jupyter serverextension enable --py jupyterlab_code_formatter

Now, if you want to test all notebooks in the :file:`examples` folder and check
how they will look like in the :ref:`contribute:Documentation`, you can do this
with:

.. code-block:: shell

  tox -e docnb

This command takes more time than :code:`tox -e doc`, but it is good practice
to do this before you submit a pull request.

.. tip::

  Sometimes it happens that your Jupyter installation does not recognize your
  :ref:`virtual environment <install:Step 2: Create a virtual environment>`. In
  that case, have a look at `these instructions
  <https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments>`__.


Spelling
--------

Throughout this repository, we follow American English (`en-us
<https://www.andiamo.co.uk/resources/iso-language-codes/>`_) spelling
conventions. As a tool, we use `cSpell
<https://github.com/streetsidesoftware/cspell/blob/master/packages/cspell/README.md>`_
because it allows to check variable names in camel case and snake case.  This
way, a spelling checker helps you in avoid mistakes in the code as well!

Accepted words are tracked through the :file:`cspell.json` file. As with the
other config files, :file:`cspell.json` formulates our conventions with regard
to spelling and can be continuously updated while our code base develops. In
the file, the :code:`words` section lists words that you want to see as
suggested corrections, while :code:`ignoreWords` are just the words that won't
be flagged. Try to be sparse in adding words: if some word is just specific to
one file, you can `ignore it inline
<https://www.npmjs.com/package/cspell#ignore>`_, or you can add the file to the
:code:`ignorePaths` section if you want to ignore it completely.

It is easiest to use cSpell in :ref:`contribute:Visual Studio Code`, through
the `Code Spell Checker
<https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker>`_
extension: it provides linting, suggests corrections from the :code:`words`
section, and enables you to quickly add or ignore words through the
:file:`cspell.json` file. Alternatively, you can `run cSpell
<https://www.npmjs.com/package/cspell#installation>`__ on the entire code base
(with :code:`cspell $(git ls-files)`), but for that your system requires `npm
<https://www.npmjs.com/>`_.


Git and GitHub
--------------

The `expertsystem` source code is maintained with Git and published through
GitHub. We keep track of issues with the code, documentation, and developer
set-up with GitHub issues (see overview `here
<https://github.com/ComPWA/expertsystem/issues>`__). This is also the place
where you can `report bugs
<https://github.com/ComPWA/expertsystem/issues/new/choose>`_.


Issue management
^^^^^^^^^^^^^^^^

We keep track of issue dependencies, time estimates, planning, pipeline
statuses, et cetera with `ZenHub <https://app.zenhub.com>`_. You can use your
GitHub account to log in there and automatically get access to the
`expertsystem` issue board once you are part of the `ComPWA organization
<https://github.com/ComPWA>`_.

Publicly available are:

* `Issue labels <https://github.com/ComPWA/expertsystem/labels>`_: help to
  categorize issues by type (maintenance, enhancement, bug, etc.).

* `Milestones
  <https://github.com/ComPWA/expertsystem/milestones?direction=asc&sort=title&state=open>`__:
  way to bundle issues for upcoming releases.


Commit conventions
^^^^^^^^^^^^^^^^^^

* Please use
  `conventional commit messages <https://www.conventionalcommits.org/>`_: start
  the commit with a semantic keyword (see e.g. `Angular
  <https://github.com/angular/angular/blob/master/CONTRIBUTING.md#type>`_ or
  `these examples <https://seesparkbox.com/foundry/semantic_commit_messages>`_,
  followed by `a column <https://git-scm.com/docs/git-interpret-trailers>`_,
  then the message. The message itself should be in imperative mood — just
  imagine the commit to give a command to the code framework. So for instance:
  :code:`feat: add coverage report tools` or :code:`fix: remove ...`.

* Keep pull requests small. If the issue you try to address is too big, discuss
  in the team whether the issue can be converted into an `Epic
  <https://blog.zenhub.com/working-with-epics-in-github>`_ and split up into
  smaller tasks.

* Before creating a pull request, run :code:`tox`. See also
  :ref:`contribute:Testing`.

* Also use a
  `conventional commit message <https://www.conventionalcommits.org/>`_ style
  for the PR title. This is because we follow a `linear commit history
  <https://docs.github.com/en/github/administering-a-repository/requiring-a-linear-commit-history>`_
  and the PR title will become the eventual commit message. Note that a
  conventional commit message style is `enforced through GitHub Actions
  <https://github.com/ComPWA/expertsystem/actions?query=workflow%3A%22PR+linting%22>`_,
  as well as :ref:`PR labels <contribute:Issue management>`.

* PRs can only be merged through 'squash and merge'. There, you will see a
  summary based on the separate commits that constitute this PR. Leave the
  relevant commits in as bullet points. See the `commit history
  <https://github.com/ComPWA/expertsystem/commits/master>`_ for examples. This
  comes in especially handy when `drafting a release <contribute:Milestones and
  releases>`_!


Milestones and releases
^^^^^^^^^^^^^^^^^^^^^^^

An overview of the `expertsystem` package releases can be found `on PyPI
history page <https://pypi.org/project/expertsystem/#history>`__. More
descriptive release notes can be found on the `release page
<https://github.com/ComPWA/expertsystem/releases>`__.

New releases are automatically published to PyPI when a new tag is created (see
`setuptools-scm <https://pypi.org/project/setuptools-scm>`_). When creating new
release notes, try to follow the style of previous releases and summarize the
changes since the last release. This can be easily seen from the linear commit
history <contribute:Git commit conventions of the master branch (see
:ref:`contribute:Commit conventions`, use :code:`git log`).


Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^

All :ref:`style checks <contribute:Automated style checks>`, testing of the
:ref:`documentation and links <contribute:Documentation>`, and :ref:`unit tests
<contribute:Testing>` are performed upon each pull request through `GitHub
Actions <https://docs.github.com/en/actions>`_ (see status overview `here
<https://github.com/ComPWA/expertsystem/actions>`__). All checks performed for
each PR have to pass before the PR can be merged.


Visual Studio code
------------------

We recommend using `Visual Studio Code <https://code.visualstudio.com/>`_ as
it's free, regularly updated, and very flexible through it's wide offer of user
extensions.

If you add or open this repository as a `VSCode workspace
<https://code.visualstudio.com/docs/editor/multi-root-workspaces>`_, the file
`.vscode/settings.json
<https://github.com/ComPWA/expertsystem/blob/master/.vscode/settings.json>`_
will ensure that you have the right developer settings for this repository. In
addition, VSCode will automatically recommend you to install a number of
extensions that we use when working on this code base (they are `defined
<https://code.visualstudio.com/updates/v1_6#_workspace-extension-recommendations>`__
:file:`.vscode/extensions.json` file).

You can still specify your own settings in `either the user or encompassing
workspace settings <https://code.visualstudio.com/docs/getstarted/settings>`_,
as the VSCode settings that come with this are folder settings.
