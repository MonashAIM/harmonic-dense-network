[TOC]
# Harmonic Densely Connected Network on medical images

## Environment setup
### 1. Dependencies Installation
#### Python
The first step is to install Python 3.11. Make sure you are installing Python 3.11 (NOT 3.12); hence, we recommend heading to the link below.

Python download link: https://www.python.org/downloads/release/python-3117/

For Mac: macOS 64-bit universal2 installer
For Windows: Windows installer (64-bit)

#### Anaconda
We  recommend installing Anaconda Navigator as this comes with VS Code and Jupyter Notebook and more importantly, the ability for you to create and manage virtual environments.

Anaconda Navigator download link: https://www.anaconda.com/download/success

#### Git
Git is a free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency

Git download link: https://git-scm.com

### 2. Cloning the repository
You should be able to clone the project repo with
```console
https://github.com/MonashAIM/harmonic-dense-network.git
```
into a designated directory (make sure it is easy to access like a folder in D or C drive)

### 3. Create new conda environment
Open **Anaconda Prompt / Anaconda Powershell Prompt** and naviagate to the folder hosting your local repo.
```console
cd PATH_TO_LOCAL_REPO
```
Run the following command
```console
conda env create -f environment.yaml
```
This steps might take awhile so no need to worry.
##Git commands
### Pull new updates
Make sure you are in **main**
```console
git checkout main
```
Fetch new changes
```console
git fetch
```
Pull new changes
```console
git pull
```

### Create a new branch
```console
git checkout -b NEW_BRANCH_NAME
```

### Push new changes
```console
git push
```
