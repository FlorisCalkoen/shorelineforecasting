# Shoreline Forecasting Data Analysis
## Forecasting Shorline Evolution Using Satellite-Derived Shoreline-Positions

Floris Calkoen, 2020, MSc Thesis Project Information Studies (University of Amsterdam) at Deltares.  

Supervised by Dr. Cristian Rodriguez Rivero (UvA), Etiënne Kras (Deltares) and Arjen Luijendijk (Deltares; Tu Delft).

## Configurations
```shell script
cd shorelineforecasting
python3.7 -m venv venv 
source venv/bin/activate 
(venv) pip install -r requirements.txt 
```

## Recommendations
Explore and understand code and date with the example notebook (example.ipynb). If you just want to run everything
at once, run ```python runner.py``` from the shorelineforecasting directory. Alternatively, run directly from root using 
```sh scripts/run.sh```

Although the framework can be run inside a Docker container, this is not recommended yet. The models are built on top of
[PyTorch](https://pytorch.org/), which makes the image extremely heavy. In near future a lighter image will be provided. 


## Citation

### Latex
```
@mastersthesis{Calkoen2020shorelines,
	author = {Calkoen, Floris R.},
	title = {Forecasting Shoreline Evolution Using Satellite-Derived Shoreline-Positions},
	year = {2020},
	note = {University of Amsterdam, The Netherlands},
}
```



 



