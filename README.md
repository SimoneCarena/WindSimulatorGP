# Run the code

- run the `set_env.sh` script to set the enviremnt variables needed to disable `numpy` multithreading

- run the `main.py` file, with the followinf possible attributes:
    - `--save_plots=all` to save the plots
    - `--show_plots=all` to show the plots
    - `--test` or `-t` to test the models. If the argument is omitted, the program will run to train the GP models
    - `--suppress_warnings` or `-w` to suppress eventual warnings

# Fans for Wind Generation

The wind field can be generated in either of the following two ways:
- Using real data collected inside a grid
- Using wind generating functions

Each of the this approach can be specified in the $\verb|.json|$ configuration file for the wind field. The configurations are mutually exclusive.

The type of the wind generation is specifed in the `"type"` field of the "`fans`" field in the configuration file, and can be `"real"` if a wind map data is provided, or `"simulated"` if the wind field is generated as a set of fan each one with its own properties.

To generate a wind field using simulated fans, one must specify, inside the `"src"` attribute, an array of fans, each with its own properites, as described in the next section.

<u>Example:</u>
```{json}
"fans": {
    "type": "simulated",
    "src": [
        {
            "x0": 0,
            "y0": 2,
            "alpha": 0,
            "noise_var": 0.1,
            "length": 4,
            "generator": {
                "function": "constant",
                "parameters": {
                    "v0": 20
                }
            }
        },
        {
            "x0": 2,
            "y0": 4,
            "alpha": -90,
            "noise_var": 0.1,
            "length": 4,
            "generator": {
                "function": "constant",
                "parameters": {
                    "v0": 20
                }
            }
        }
    ]
```

To use a precomputed wind map, in the `"src"` field, two fields have to be specified:
- the location of the wind field mean map, inside the `"mean"` field
- the location of the wind field variance map, inside the `"var"` field
- a scale factor `scale_factor` to modulate the wind speed

## Function Generator

$\verb|json|$ formatting to generate wind-generating functions. The function description field is called $\verb|"generator"|$, and is one of the fan's fields. In the examples reported here, only the $\verb|"generator"|$ field of the fan is reported.

To each function $v_0(t)$ is also added a Gaussian noise $\mathcal{N}(0,\sigma_n^2)$, where $\sigma_n^2$ is specified by the `nosie_var` parameter in the $\verb|json|$ file, and represents the variability in the generated wind.

### $\sin$ and $\cos$ Functions
`"function"` parameter name: `"sin"`, `"cos"`

$$
v_0(t)=\frac{V_0}{2}\sin(2\pi ft+\phi_0)+\frac{V_0}{2}
$$

$$
v_0(t)=\frac{V_0}{2}\cos(2\pi ft+\phi_0)+\frac{V_0}{2}
$$

Function `"parameters"`:
- $f$: `"frequency"` parameter
- $\phi_0$: `"phase"` parameter
- $V_0$: `"v0"` parameter

<u>Example:</u>
```{json}
"generator": {
    "function": "sin",
    "parameters": {
        "v0": 15,
        "frequency": 0.01,
        "phase": 0.0
    }
}
```
```{json}
"generator": {
    "function": "cos",
    "parameters": {
        "v0": 10,
        "frequency": 0.5,
        "phase": 30.0
    }
}
```

### Square Function
`"function"` parameter name: `"square"`

$$
    v_0(t)=\frac{V_0}{2}\,\text{sign}\left(\sin\left( 2\pi ft\right)\right)+\frac{V_0}{2}
$$

Function `"parameters"`:
- $f$: `"frequency"` parameter
- $V_0$: `"v0"` parameter

<u>Example:</u>
```{json}
"generator": {
    "function": "square",
    "parameters": {
        "v0": 12,
        "frequency": 0.1,
    }
}
```

### Constant Function
$\verb|"function"|$ parameter name: $\verb|"constant"|$

$$
    v_0(t)=V_0
$$

Function `"parameters"`:
- $V_0$: `"v0"` parameter

<u>Example:</u>
```{json}
"generator": {
    "function": "constant",
    "parameters": {
        "v0": 10,
    }
}
```

## Fan Generation Function

$$
v(t,d_\perp,d_\parallel) = \frac{v_0(t)}{2}\frac{\tanh\left( \frac{d_\perp+\frac{L}{2}}{w\cdot \left(d_\parallel+1\right)} \right)-\tanh\left(\frac{d_\perp-\frac{L}{2}}{w\cdot \left(d_\parallel+1\right)}\right)}{d_\parallel+1}
$$

where:
- $v_0(t)$ is the velocity in the center, as defined before
- $L$ is the width of the fan
- $w = 0.002 $ determines the spread of the wind cone
- $d_\perp$ and $d_\parallel$ are the distances between the center of the fan and the point $\mathbf{p}=[x,y]^{\text{T}}$ along, respectively, the axis perpendicular to the wind direction versor $\mathbf{u}_0$ and the axis parallel to it. Such distances are computed as

$$
d_\parallel=\left| \mathbf{p}\cdot\mathbf{u}_0 \right|
$$

$$
d_\perp=\left| \mathbf{p}\cdot\mathbf{u}_0^\perp \right|
$$

!['Wind Speed'](imgs/readme/fan_xyz.png)

## Real Wind Field

To use a wind field consisting of real world measurements, the `"type"` parameter has to be `"real"`, and the `"src"` parameter must contain the `"mean"` and `"var"` fields, specifying the location of the files for the, respectively, mean and variance wind map.

Each file has to be a $\verb|numpy|$ file, i.e. a file with $\verb|.npy|$ format, containing an $N\times N\times 3$ matrix of wind measurements. 

To compute the wind speed at position $(x,y)$ the following operations are performed:
- The position is translated into a pair of indices $(i,j)$ for the grid, as

    - $\verb|i| = x\cdot\frac{ \verb|grid_resolution|}{\verb|grid_width|}$

    - $\verb|j| = y\cdot\frac{\verb|grid_resolution|}{\verb|grid_height|}$

- The speed is drawn from a multivariate normal, with mean $\boldsymbol{\mu^{V}}=[\verb|mean_map[i,j,0]|,\verb|mean_map[i,j,1]|]^{\text{T}}$ and variance $\boldsymbol{\Sigma^{V}}=\text{diag}\left([\verb|var_map[i,j,0]|,\verb|var_map[i,j,1]|]\right)$, as in

$$
    \mathbf{V}\sim\mathcal{N}\left(\boldsymbol{\mu^V},\boldsymbol{\Sigma^{V}}\right)
$$

In case a scale factor `scale_factor` is used, the resulting wind speed will be generated as

$$
    \mathbf{V}\sim\mathcal{N}\left(\verb|scale_factor|\cdot\boldsymbol{\mu^V},\boldsymbol{\Sigma^{V}}\right)
$$