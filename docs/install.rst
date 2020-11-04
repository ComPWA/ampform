Installation
============

The fastest way of installing the `expertsystem` is through PyPI:

.. code-block:: shell

  python3 -m pip install expertsystem

This installs the `latest release <https://pypi.org/project/expertsystem>`_
that you can find on the `stable
<https://github.com/ComPWA/expertsystem/tree/stable>`_ branch. The latest
version on the `master <https://github.com/ComPWA/expertsystem/tree/master>`_
branch can be installed as follows:

.. code-block:: shell

  python3 -m pip install git+https://github.com/ComPWA/expertsystem@master

but in that case, we highly recommend using the more dynamic,
:ref:`'editable mode' <install:Editable mode>` instead.


Editable mode
-------------

The `expertsystem` is an academic research project and is bound to continuously
evolve. We therefore highly recommend installing the `expertsystem` from `the
source code <https://github.com/ComPWA/expertsystem>`_ as an `editable install
<https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_, so
that you work with the latest version and try out your own modifications to the
source code.

Moreover, since you read as far as this, you must have an interest in particle
physics, and it is researchers like you who can help bring this project
further! So please, follow the following sections to set up this 'interactive
installation'.


.. _local-repository:

Step 1: Get the source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `expertsystem` source code is maintained through `Git
<https://git-scm.com>`_, so you need to `install Git
<https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_ first. Once
you've done so, navigate to a suitable folder and run:

.. code-block:: shell

  git clone https://github.com/ComPWA/expertsystem
  cd expertsystem

After that, there should be a folder called :file:`expertsystem` into which we
navigated just now. We'll call this folder the **local repository**.

When new commits are merged into the `master branch of expertsystem
<https://github.com/ComPWA/expertsystem/tree/master>`_, you need to update your
local copy of the source code as follows:

.. code-block:: shell

  git checkout master
  git pull

It's best to have a clean your `working tree
<https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository>`_
before you do a :command:`git pull`.


Step 2: Create a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is safest to install the `expertsystem` within a virtual environment, so
that all Python dependencies are contained within there. This is helpful in
case something goes wrong with the dependencies: you can just trash the
environment and recreate it. There are two options: Conda or Python's venv.

.. tabbed:: Conda environment

  `Conda <https://www.anaconda.com/>`_ can be installed without administrator
  rights, see instructions on `this page
  <https://www.anaconda.com/distribution/>`_. Once installed, navigate to the
  :ref:`local repository <local-repository>` and create the Conda environment
  for the `expertsystem` as follows:

  .. code-block:: shell

    conda env create

  This command uses the `environment.yml
  <https://github.com/ComPWA/expertsystem/blob/master/environment.yml>`_ file
  and immediately installs the `expertsystem` in `editable mode
  <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`__.

  After Conda finishes creating the environment, you can activate it with as
  follows:

  .. code-block:: shell

    conda activate es

  You need to have the environment called :code:`es` activated whenever you
  want to run the `expertsystem`.


.. tabbed:: Python venv

  Alternatively, you can use `Python's venv
  <https://docs.python.org/3/library/venv.html>`_, if you have that available
  on your system. All you have to do, is navigate into :ref:`local repository
  <local-repository>` and run:

  .. code-block:: shell

    python3 -m venv ./venv

  This creates a folder called :file:`venv` where all Python packages will be
  contained. You first have to activate the environment, and will have to do so
  whenever you want to run the `expertsystem`.

  .. code-block:: shell

    source ./venv/bin/activate

  Now you can safely install the `expertsystem` in `editable mode
  <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`__:

  .. code-block:: shell

    pip install -e .

That's it, now you're all set to :doc:`help develop the project <develop>`!


Step 3: Test the installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you've installed the `expertsystem`, simply launch a Python interpreter
and run:

.. code-block:: python

  import expertsystem

If you don't get any error messages, all worked out nicely!

For more thorough testing, navigate back to the you can run the unit tests:

.. code-block:: shell

  pip install -e .[test]  # install dependencies for testing
  pytest -n auto

After that, it's worth having a look at the :doc:`contribute page <develop>`!

Updating to the latest version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When new commits are merged into the `master branch
<https://github.com/ComPWA/expertsystem/tree/master>`_, you need to update your
:ref:`local copy of the source code <local-repository>` as follows:

.. code-block:: shell

  git checkout master
  git pull
  pip install -e .

It's best to have a clean your `working tree
<https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository>`_
before you do a :command:`git pull`. We also call :command:`pip install` again,
because we sometimes introduce upgrades of the dependencies.

If you face any issues when calling :code:`pip install -e .`, just trash your
install Conda environment or venv and repeat from :ref:`Step 2 <install:Step 2:
Create a virtual environment>`.
