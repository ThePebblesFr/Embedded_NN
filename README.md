# Embedded NN Project

|               |                                 |
|---------------|---------------------------------|
|Authors        |Mickaël JALES, Pierre GARREAU    |
|Status         |Under development                |
|Description    |Embedded Neural Network project dealing with grapevine leaves dataset for early detection and classification of esca disease in vineyards. This code is meant to be executed on STM32L439 board |
|Project        |ISMIN 3A - Embedded IA           |

# Table of contents


# Commands to execute

## We advise you to create a virtual environment to work on the project. To do this, create an environment at the root of the project with the following commands: 

python -m venv envML
<!-- if your are on Windows -->
envML/Scripts/activate      <!-- allows to use the virtual python working environment -->
<!-- otherwise -->
source envML/bin/activate

deactivate                  <!-- disable the environment >

<!-- if you have a policy issue with powershell or windows-->
Set-ExecutionPolicy -Scope "CurrentUser" -ExecutionPolicy "Unrestricted" <!-- to disable the restrictions -->

Set-ExecutionPolicy -Scope "CurrentUser" -ExecutionPolicy "RemoteSigned" <!-- to enable the restrictions -->


## install the necessary packages in your own virtual environment (this can also works on your defaut environment)
pip install -r requirements.txt <!--or--> python -m pip install -r requirements.txt

## in .gitignore, the python virtual environment is ignored
<!-- it allows you to use the python working environment with all necessary packages only on your computer -->
