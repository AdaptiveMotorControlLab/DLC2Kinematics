

# How to Contribute to DLC2Kinematics

DLC2Kinematics is an actively developed package and we welcome community development and involvement.
We are especially seeking people from underrepresented backgrounds in OSS to contribute their expertise and experience.
Please get in touch if you want to discuss specific contributions you are interested in developing, and we can help shape a road-map.

**Please note, this code is available for non-commericial use only, and you will need to please
e-mail a signed copy of the CLAI (and if applicable the CLAC) as PDF file to mackenzie.mathis@epfl.ch before a PR would be accepted.**

We are happy to receive code extensions, bug fixes, documentation updates, etc.

If you are a new user, we recommend checking out the detailed [Github Guides](https://guides.github.com).

## Setting up a development installation

In order to make changes to `dlc2kinematics`, you will need to [fork](https://guides.github.com/activities/forking/#fork) the
[repository](https://github.com/adaptivemotorcontrollab/DLC2Kinematics).

If you are not familiar with `git`, we recommend reading up on [this guide](https://guides.github.com/introduction/git-handbook/#basic-git).

Here are guidelines for installing deeplabcut locally on your own computer, where you can make changes to the code!
We often update the main kinemaitk code base on github, and then we push out a stable release on pypi.
This is what most users turn to on a daily basis (i.e. pypi is where you get your `pip install dlc2kinematics` code from!

But, sometimes we add things to the repo that are not yet integrated, or you might want to edit the code yourself,
or you will need to do this to contribute. Here, we show you how to do this.

**Step 1:**

- git clone the repo into a folder on your computer:  

- click on this green button and copy the link.

- then in the terminal type: `git clone https://github.com/adaptivemotorcontrollab/DLC2Kinematics.git`

**Step 2:**

- Now you will work from the terminal inside this cloned folder.

- Now, when you start `ipython` and `import dlc2kinematics` you are importing the folder "DLC2Kinematics" - so any changes you make, or any changes we made before adding it to the pip package, are here.

- You can also check which DLC2Kinematics you are importing by running: `dlc2kinematics.__file__`

If you make changes to the code/first use the code, be sure you run `./resinstall.sh`, which you find in the main folder.


Note, before committing to DLC2Kinematics, please be sure your code is formatted according to `black`. To learn more,
see [`black`'s documentation](https://black.readthedocs.io/en/stable/).

Now, please make a pull request that includes both a **summary of and changes to**:

- How you modified the code and what new functionality it has.
- DOCSTRING update for your change
- A working example of how it works for users.
- If it's a function that also can be used in downstream steps (i.e. could be plotted)
we ask you (1) highlight this, and (2) idealy you provide that functionality as well.

**Review & Formatting:**

- Please run black on the code to conform to our Black code style (see more at https://pypi.org/project/black/).
- Please assign a reviewer, typically @mmathislab.
