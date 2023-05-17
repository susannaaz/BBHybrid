# BBHybrid

Used to calculate a set of well-educated constant spectral indices βS and βD and clean out the
spatially-constant part of the foregrounds at the map-level. In practice, a filter Q is returned
that de-projects the linear combinations of the data that follow the best-fit SEDs of the
foreground sources under consideration (dust and synchrotron).

The outputs are directly compatible with the inputs to [BBPipe](https://github.com/simonsobs/BBPipe),
its submodule [BBPower](https://github.com/simonsobs/BBPower). 
The inputs can be generated with [BBSims](https://github.com/susannaaz/BBSims).

Ref:
[A hybrid map-Cℓ  component separation method for primordial CMB B-mode searches](https://iopscience.iop.org/article/10.1088/1475-7516/2023/03/035)

Developed by: Susanna Azzoni, Max Abitbol, David Alonso