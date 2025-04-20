# Joint Probability Visualizer
## Summary

This program allows users to create interactive 3D visualizations of joint probability distributions. Users can select from five different types of distributions:
* Normal
* Exponential
* Gamma
* Beta
* Uniform

Users can customize the parameters of these different distributions. 

The program generates a 3D surface plot where the $x$ and $y$ axes represent the two selected distributions and the $z$-axis represents their joint probability density. Visualized using a rainbow gradient, colors closer to blue or purple represent lower density values, while colors closer to red represent higher density values.

## Required Libraries
For this program to work correctly, users need to have Python 3.10 installed along with:
* `numpy`
* `matplotlib`
* `scipy`

  All of which can be installed using `pip install numpy matplotlib scipy` in the terminal.
