# Fusion Config Options

<pre data-full-width="true"><code>projection_id (int): ID of the projection used.

description (str): Description of the fusion process.

area_id (str): Identifier for the area.

projection_proj4: Projection details

<strong>    proj (str): Projection type.
</strong>
    datum (str): Datum used for projection.

final_resolution (float): Final resolution of the fusion process.

projection_units (str): Units used in projection.

resample_radius (int): Radius for resampling.

resample_n_neighbors (int): Number of neighbors for resampling.

resample_n_procs (int): Number of processes used for resampling.

resample_epsilon (float): Epsilon value used in resampling.

use_bilinear (bool): Boolean value indicating whether or not bilinear interpolation is used.
</code></pre>
