# Function Generator

$\verb|json|$ formatting to generate wind-generating functions. The function description field is called $\verb|"generator"|$, and is one of the fan's fields. In the examples reported here, only the $\verb|"generator"|$ field of the fan is reported.

## $\sin$ and $\cos$ Functions
$\verb|"function"|$ parameter name: $\verb|"sin"|$, $\verb|"cos"|$
$$
v_0(t)=\frac{V_0}{2}\sin(2\pi ft+\phi_0)+\frac{V_0}{2}
$$
$$
v_0(t)=\frac{V_0}{2}\cos(2\pi ft+\phi_0)+\frac{V_0}{2}
$$
Function $\verb|"parameters"|$:
- $f$: $\verb|"frequency"|$ parameter
- $\phi_0$: $\verb|"phase"|$ parameter
- $V_0$: $\verb|"v0"|$ parameter

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

## Square Function
$\verb|"function"|$ parameter name: $\verb|"square"|$
$$
    v_0(t)=\frac{V_0}{2}\,\text{sign}\left(\sin\left( 2\pi ft\right)\right)+\frac{V_0}{2}
$$
Function $\verb|"parameters"|$:
- $f$: $\verb|"frequency"|$ parameter
- $V_0$: $\verb|"v0"|$ parameter

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

## Constant Function
$\verb|"function"|$ parameter name: $\verb|"constant"|$
$$
    v_0(t)=V_0
$$
Function $\verb|"parameters"|$:
- $V_0$: $\verb|"v0"|$ parameter

<u>Example:</u>
```{json}
"generator": {
    "function": "constant",
    "parameters": {
        "v0": 10,
    }
}
```

# Fan Generation Function

$$

v(x,y,t) = \frac{v_0(t)}{2}\frac{\tanh\left( \frac{d_\perp+\frac{L}{2}}{w|d_\parallel|} \right)-\left(\frac{d_\perp-\frac{L}{2}}{w|d_\parallel|}\right)}{\left| d_\parallel \right|+1}

$$
where:
- $v_0(t)$ is the velocity in the center, as defined before
- $L$ is the width of the fan
- $w = 0.002 $