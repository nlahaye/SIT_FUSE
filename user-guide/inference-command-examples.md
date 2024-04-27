# Inference Command Examples

Make sure config files have been updated to contain all desired paths and parameterizations.

{% code fullWidth="true" %}
```
cd <path_to_SIT_FUSE>/SIT_FUSE/sit_fuse/inference/
#Can be run outside of the repo via command line or in a scripted fashion as well
```
{% endcode %}

```
python3 generate_output.py -y <path_to_yaml>
#i.e. set <path_to_yaml> to ../config/model/emas_fire_dbn_multi_layer_pl.yaml 
```

