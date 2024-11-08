---
description: >-
  Shape files (.shp) are QGIS's way of storing location, shape, and attributes
  of geographic features (aka labels). Continuing our example, we'll take a look
  at how to create shape files in QGIS.
icon: fill-drip
---

# Creating and modifying shapefiles

1. Open QGIS.
2. Open the GeoTiff file(s) to be labeled.
3.  Open the [Bing Satellite](qgis-configuration.md).



    <figure><img src="../../../.gitbook/assets/Screenshot 2024-08-13 at 12.11.09 PM.png" alt=""><figcaption><p>Example of step 3</p></figcaption></figure>
4. Make sure the [Data Source Manager Toolbar](qgis-configuration.md) is on your screen.
5.  In the Data Source Manager Toolbar at the top, select the icon with four dots that says `New Shapefile Layer...` when hovered over.



    <figure><img src="../../../.gitbook/assets/Screenshot 2024-08-15 at 12.00.55 PM.png" alt=""><figcaption><p>New Shapefile Layer</p></figcaption></figure>



    <figure><img src="../../../.gitbook/assets/Screenshot 2024-10-18 at 12.46.43 PM.png" alt="" width="375"><figcaption><p>New Shapefile Layer menu</p></figcaption></figure>
6. In the window that appears begin by giving the file a name next to `File name`.&#x20;

Try to give it a unique name that won't be confused later. For example, include the name of the instrument, the date that the data corresponds to, and the feature being labeled (eMAS\_08022019\_fire). This is just a guideline though, feel free to be as detailed as you'd like. Naming the shapefile after the original file with the specific feature at the end is also always a safe bet (eMASL1B\_19910\_06\_20190806\_1815\_1824\_V03\_fire).

7. Then, select the button with 3 dots to the right of the file name, choose a directory for the shapefile to be stored in, and click save.&#x20;

Try to save all the shape files for one image/images from the same instrument/scene in the same folder for easier access later. Stay consistent with how/where shape files (for every image) are stored, we'll need to access their file paths later.

**Labeler's Note**: I would recommend the example organization below. Other organizations (such as folders corresponding to each label\_type (e.g. smoke, fire, background, etc.)) typically become a lot harder to work with during future steps when we need to gather all of the files using the command line. An organization, with each folder corresponding to an image, makes organizing the YAML a lot easier.

Here's an example organization:

<figure><img src="../../../.gitbook/assets/Screenshot 2024-10-24 at 7.08.28 PM.png" alt=""><figcaption><p>Organized GeoTiffs and labels. The second column from the left is all of the GeoTiffs (.tif) images (which are all the same scene but at different days/times), each folder in the third column corresponds to one GeoTiff in the second (one to one matching), and the fourth column contains all of the shape files (and associated files) for the specific GeoTiff image that it's labels are for.</p></figcaption></figure>

8. For Geometry type, select Polygon.&#x20;

This allows us to have a 2D shape rather than just a line.

9. At the bottom right of the window click OK.&#x20;

_Note: if you'd like to change the color of the shapefile, select the shapefile from the Layers panel and hover over to the Layer Styling panel and select any color._

Congrats, you now have your first shape file! Labeling is pretty simple.

1. With the data layer selected (in the [Layers panel](qgis-configuration.md)), navigate to the [Layer Styling panel](qgis-configuration.md) and choose which spectral band is best for viewing the feature you are labeling. For more documentation on this, see [Understanding the data](understanding-the-data.md).
2. Now, with the shapefile layer selected, select the yellow pencil button from the [Digitizing toolbar](qgis-configuration.md) that says `Toggle Editing` when hovered over.
3. Then select the green button to the right of the pencil (that appears when the pencil is selected) that says `Add Polygon Feature` when hovered over.&#x20;

<figure><img src="../../../.gitbook/assets/Screenshot 2024-10-18 at 1.07.14 PM.png" alt=""><figcaption><p>Toolbars</p></figcaption></figure>

3. Click on the perimeter of the feature you would like to label and begin labeling.
4. Once you have completed your label, right click on the screen and select OK in the pop up that appears.&#x20;

Here is an example of a label of the burnscar from the William's Flat's Fire.

<figure><img src="../../../.gitbook/assets/Screenshot 2024-10-18 at 1.18.57 PM.png" alt="" width="375"><figcaption><p>Figure 1. eMAS William's Flat's Fire, greyband 9, burnscar label</p></figcaption></figure>

Congrats, you now have your first label! Use these steps to create as many labels as needed. For more documentation on more complex features (deleting labels, modifying labels, etc.) see [Other shapefile tools](other-shapefile-tools.md).

**Labeler's Note:** For better results, create a background label(s)! You can create a general label (e.g. \_background_)_ for each image that encompasses everything _except_ the features you are labeling, or you can create a background label for each specific feature (e.g. \_fire_\__background, \_smoke_\__background) of each image, which encompasses everything _except_ that specific feature. For example, its okay if the label for \_fire_\__background overlapped onto the \_smoke label.
