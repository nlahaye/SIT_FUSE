---
description: >-
  QGIS is a geographic information system software that supports the viewing,
  editing, and analysis of geospatial data in a range of data formats. It can be
  used to create labels for SIT-FUSE.
---

# QGIS configuration

1. **Download QGIS**

{% embed url="https://qgis.org/download/" %}
MacOS users: Note the "Tips for first launch" Section
{% endembed %}

2.  I**nstall Plugin(s)**

    1. Open QGIS.
    2. Select `Plugins -> Manage and Install Plugins` from the menu bar at the top.
    3. Type and select `QuickMapServices` on the `All` page.
    4. Select `Install Plugin` at the bottom.
    5. Select `Web -> QuickMapServices -> Settings` from the menu bar at the top.
    6. Select the `More services` tab.
    7. Click `Get contributed pack`.

    You can now select `Web -> QuickMapServices -> Bing -> Bing Satellite` from the menu bar at the top and have a world map, a useful resource for labeling.
3.  S**et up Toolbars/Panels**

    1. Open QGIS.
    2. Select `View -> Panels -> Layer Styling` from the menu bar at the top to modify layer visibility features and settings.
    3. Select `View -> Toolbars -> Digitizing Toolbar` and `View -> Toolbars -> Advanced Digitizing Toolbar` from the menu bar at the top to access key shape file modification features.
    4. Select `View -> Toolbars -> Data Source Manager Toolbar` from the menu bar at the top to access layer creation features.
    5. Select `View -> Panels -> Layers` from the menu bar at the top to see various layers (e.g. the data being labeled, various shape files, etc.)

    _Note: if there's already a check mark next to the tool/panel, no need to select it again._

These are just a few of the many tools that can be useful for labeling. Further documentation about other tools can be found in [here](https://docs.qgis.org/2.18/en/docs/user_manual/index.html) and in the next few GitBook pages.
